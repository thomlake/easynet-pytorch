#
# Parameter initialization
#

import math
import numpy as np
import torch


def glorot(w):
    """Glorot (aka Xavier) initialization.
    
    title: Understanding the difficulty of training deep feedforward neural networks
    paper: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    n_in, n_out = w.size()
    b = math.sqrt(6) / math.sqrt(n_in + n_out)
    return w.uniform_(-b, b)


def orthonormal(w, gain=math.sqrt(2)):
    """Orthonormal initialization.
    
    title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    paper: http://arxiv.org/pdf/1312.6120v3.pdf
    """
    n_in, n_out = w.size()
    n = max(n_in, n_out)
    m = np.random.normal(0, 1, (n, n))
    m = np.linalg.svd(m)[0][:n_in, :n_out]
    m *= gain
    m = torch.from_numpy(m)
    return w.copy_(m)


def spectral(w, s=1.0):
    """Spectral initialization.

    Initialize from a normal distribution and scale such that the spectral radius is likely s.
    http://danielrapp.github.io/rnn-spectral-radius/
    """
    n_in, n_out = w.size()
    n = max(n_out, n_in)
    gain = s / math.sqrt(n)
    return w.normal_(0, 1).mul_(gain)


def sparse(w, pz=0.9, gain=None):
    """Sparse initialization
    
    title: On the importance of initialization and momentum in deep learning
    paper: http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """
    n_in, n_out = w.size()
    gain = gain or math.sqrt(6) / math.sqrt(n_in + n_out)
    
    m = gain * np.random.normal(0, 1, (n_in, n_out))
    nz = min(n_out - 1, int(round(pz * n_out)))
    js = np.arange(n_out)

    for i in range(n_in):
        np.random.shuffle(js)
        m[i, js[:nz]] = 0
    
    dead_out = np.isclose(np.abs(m).sum(0), 0).nonzero()[0]
    for j in dead_out:
        i = np.random.randint(0, n_in)
        m[i, j] = gain * np.random.normal(0, 1)

    if np.any(np.isclose(np.abs(m).sum(1), 0)):
        raise RuntimeError('dead inputs')

    if np.any(np.isclose(np.abs(m).sum(0), 0)):
        raise RuntimeError('dead outputs')

    m = torch.from_numpy(m)
    return w.copy_(m)
