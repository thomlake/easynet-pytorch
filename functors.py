"""Various neural network functors."""

import torch
from torch import DoubleTensor, FloatTensor
from torch.nn import functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList, Tanh

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

    def forward(self, args):
        x, h = args
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
            funcnet.functions.expand_leading_as(self.h0, x),
            funcnet.functions.expand_leading_as(self.c0, x))

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
