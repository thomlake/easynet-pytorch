#
# Probabilistic losses
#

import math

from torch.nn import functional as F
from easynet.functions import mse


class normal(object):
    @staticmethod
    def predict(x):
        return x

    @staticmethod
    def cost(p, y, size_average=True):
        return mse(p, y, size_average=size_average)


class cat(object):
    @staticmethod
    def probs(x):
        return F.softmax(x)
    
    @staticmethod
    def predict(x):
        return x.max(1)[1]
    
    @staticmethod
    def cost(x, y):
        return F.cross_entropy(x, y)


class bern(object):
    @staticmethod
    def probs(x):
        return F.sigmoid(x)
    
    @staticmethod
    def predict(x, threshold=0.5):
        threshold = math.log(threshold / (1 - threshold))
        return (x.data > threshold).float()
    
    @staticmethod
    def cost(x, y):
        return F.binary_cross_entropy(bern.probs(x), y)