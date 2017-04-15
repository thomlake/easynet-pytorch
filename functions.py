#
# Extra functions
#

import torch
import torch.nn.functional as F


def expand_leading_as(x, y):
    return x.expand(*(y.size()[:-1] + (x.size(-1),)))


def mse(input, target, size_average=True):
    return torch.nn._functions.thnn.MSELoss(size_average)(input, target)


def inner_prod(x, y):
    """Compute the inner products between rows in x and y."""
    return (x * y).sum(1)


def cosine_sim(x, y):
    """Compute the cosine similarity between rows of x and y."""
    Z = x.norm(dim=1) * y.norm(dim=1)
    return inner_prod(x, y) / Z


def weighted_sum(weights, values):
    """Compute the sum of weights[i] * values[i].

    Expands weights[i] to match values[i].

    Raises:
        ValueError: if len(weights) != len(values)
        ValueError: if len(weights) == len(values) == 0
        ValueError: if weights[i].expand_as(values[i]) fails
        RuntimeError: if all weights[i] * values[i] can not be summed 
    """
    if len(weights) != len(values):
        raise RuntimeError('len(weights) = {} != {} = len(values)'.format(len(weights), len(values)))
    if len(weights) == 0:
        raise RuntimeError('cannot computed weighted sum of empty sequences')
    
    return sum(w.expand_as(v) * v for w, v in zip(weights, values))


def attend(query, keys, values=None, score=None):
    """Apply softmax attention over keys.

    Args:
        query: the query to compare to each key.
        keys: an iterable of items to compare to query.
        values: if given returns the weighted sum of probs[i] * values[i].
        score: the function used to compare each (key, query) pair.

    Returns:
        probs if values is None, otherwise the weighted sum of probs[i] * values[i].
    
    Raises:
        ValueError: if len(keys) < 1
        See weighted_sum for other potential errors.
    """
    n_keys = len(keys)
    if n_keys < 1:
        raise RuntimeError('cannot compute attention over empty sequence')

    score = score or inner_prod
    scores = [score(key, query) for key in keys]
    probs = F.softmax(torch.cat(scores, 1)).chunk(n_keys, 1)
    
    if values is None:
        return probs
    return weighted_sum(probs, values)


def scan(f, xs, initial=None, observe=True):
    s = initial or f.initial(xs[0])
    ys = []
    for x in xs:
        s = f(x, s)
        ys.append(f.observe(s) if observe else s)
    return ys
