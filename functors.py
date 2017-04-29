"""Various neural network functors."""

import torch
from torch import DoubleTensor, FloatTensor
from torch.nn import functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList, ReLU, Sigmoid, Tanh

from easynet import functions, init


class Affine(Module):
    def __init__(self, w, b):
        super(Affine, self).__init__()
        self.w = Parameter(w.t())
        self.b = Parameter(b)

    def forward(self, x):
        return self._backend.Linear()(x, self.w, self.b)


def affine(n_feat, n_out):
    w = init.glorot(FloatTensor(n_feat, n_out))
    b = FloatTensor(n_out).zero_()
    return Affine(w, b)


class MultiAffine(Module):
    def __init__(self, ws, b):
        super(MultiAffine, self).__init__()
        self.w = Parameter(torch.cat(ws, 0).t())
        self.b = Parameter(b)

    def forward(self, x):
        x = torch.cat(x, 1)
        return self._backend.Linear()(x, self.w, self.b)


def multi_affine(n_feats, n_out):
    ws = [init.glorot(FloatTensor(n_feat, n_out)) for n_feat in n_feats]
    b = FloatTensor(n_out).zero_()
    return MultiAffine(ws, b)


class Recurrent(Module):
    def observe(self, state):
        return state

    def initial(self, x):
        return functions.expand_leading_as(self.initial_state, x)


class StackRecurrent(Module):
    def __init__(self, layers):
        super(StackRecurrent, self).__init__()
        self.layers = ModuleList(layers)

    def initial(self, x):
        return [layer.initial(x) for layer in self.layers]

    def observe(self, states):
        return self.layers[-1].observe(states[-1])

    def forward(self, x, states_prev):
        states_curr = []
        for i, (layer, state) in enumerate(zip(self.layers, states_prev)):
            state = layer(x, state)
            states_curr.append(state)
            x = layer.observe(state)
        return states_curr


def stack_recurrent(sizes, rnn=None):
    if len(sizes) < 2:
        raise ValueError('len(sizes) must be > 2 (got: {})'.format(sizes))
    rnn = rnn or elman
    return StackRecurrent([rnn(n_feat, n_hid) for n_feat, n_hid in zip(sizes, sizes[1:])])


class Elman(Recurrent):
    def __init__(self, f, g, initial_state=None):
        super(Elman, self).__init__()
        self.f = f
        self.g = g
        if initial_state is not None:
            self.initial_state = Parameter(initial_state)

    def forward(self, x, h):
        return self.f(self.g([x, h]))


def elman(n_feat, n_hid, f=None, initial_state=True):
    w = init.glorot(FloatTensor(n_feat, n_hid))
    u = init.orthonormal(FloatTensor(n_hid, n_hid), gain=1)
    b = FloatTensor(n_hid).zero_()
    initial_state = FloatTensor(n_hid).zero_() if initial_state else None

    f = f or Tanh()
    g = MultiAffine([w, u], b)

    return Elman(f, g, initial_state)


class GRU(Recurrent):
    def __init__(self, f_ru, f_c, initial_state=None):
        super(GRU, self).__init__()
        self.f_ru = f_ru
        self.f_c = f_c
        if initial_state is not None:
            self.initial_state = Parameter(initial_state)

    def forward(self, x, h):
        r, u = torch.chunk(F.sigmoid(self.f_ru([x, h])), 2, dim=1)
        c = F.tanh(self.f_c([x, r * h]))
        return u * h + (1 - u) * c


def gru(n_feat, n_hid, initial_state=True):
    w_ru = init.glorot(FloatTensor(n_feat, 2 * n_hid))
    u_ru = init.orthonormal(FloatTensor(n_hid, 2 * n_hid), gain=1)
    b_ru = torch.cat([FloatTensor(n_hid).fill_(1), FloatTensor(n_hid).fill_(0)])
    
    f_ru = MultiAffine([w_ru, u_ru], b_ru)
    w_c = init.glorot(FloatTensor(n_feat, n_hid))
    u_c = init.orthonormal(FloatTensor(n_hid, n_hid), gain=1)
    b_c = FloatTensor(n_hid).fill_(0)
    f_c = MultiAffine([w_c, u_c], b_c)
    
    initial_state = FloatTensor(n_hid).zero_() if initial_state else None
    
    return GRU(f_ru, f_c, initial_state)


class LSTM(Recurrent):
    def __init__(self, f, initial_state=None):
        super(LSTM, self).__init__()
        self.f = f
        if initial_state is not None:
            h0, c0 = initial_state
            self.h0 = Parameter(h0)
            self.c0 = Parameter(c0)
    
    def initial(self, x):
        return (
            functions.expand_leading_as(self.h0, x),
            functions.expand_leading_as(self.c0, x))

    def observe(self, state):
        return state[0]
    
    def forward(self, x, state):
        h_prev, c_prev = state
        i, f, o, g = torch.chunk(self.f([x, h_prev]), 4, dim=1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        o = F.sigmoid(o)
        g = F.tanh(g)
        c = f * c_prev + i * g
        h = o * c
        return h, c


def lstm(n_feat, n_hid, initial_state=True):
    w = init.glorot(FloatTensor(n_feat, 4 * n_hid))
    u = init.orthonormal(FloatTensor(n_hid, 4 * n_hid), gain=1)
    b = torch.cat([FloatTensor(n_hid).fill_(1 if i == 1 else 0) for i in range(4)])
    f = MultiAffine([w, u], b)
    
    initial_state = (FloatTensor(n_hid).zero_(), FloatTensor(n_hid).zero_()) if initial_state else None
    
    return LSTM(f, initial_state)


class Multiplicative(Recurrent):
    """Multiplicative Integration RNN.

    On Multiplicative Integration with Recurrent Neural Networks
    https://arxiv.org/pdf/1606.06630.pdf
    """
    def __init__(self, w, u, a, b, c, d, initial_state, f=None):
        super(Multiplicative, self).__init__()
        self.f = f or Tanh()
        self.w, self.u = w, u
        self.a, self.b, self.c = a, b, c
        self.d = d
        self.initial_state = initial_state

    def forward(self, x, h):
        wx = x.mm(self.w)
        uh = h.mm(self.u)
        g1 = self.a.expand_as(wx) * wx * uh
        g2 = self.b.expand_as(wx) * wx + self.c.expand_as(uh) * uh
        h = self.f(g1 + g2 + self.d.expand_as(g1))
        return h


def multiplicative(n_feat, n_hid, f=None):
    w = Parameter(init.glorot(FloatTensor(n_feat, n_hid)))
    u = Parameter(init.orthonormal(FloatTensor(n_hid, n_hid), gain=1))
    a, b, c, d, h0 = [Parameter(FloatTensor(n_hid).zero_()) for i in range(5)]
    return Multiplicative(w, u, a, b, c, d, h0, f=f)

