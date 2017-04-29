import numpy as np
import time
import torch

from collections import defaultdict
from torch.optim import *
from torch.nn.utils import clip_grad_norm


def add_grad_noise(parameters, scale=0.01):
    for p in parameters:
        p.grad.data.add_(torch.randn(p.size()).mul_(scale))


class TrainingMonitor(object):
    """Utility class to monitor training progress.

    Usage
    -----
        monitor = TrainingMonitor('nll')
        while not train_monitor.converged:
            nll, train_error, valid_error = train_step(net, train_data, valid_data)
            train_monitor.observe(nll=nll, train_error=train_error, valid_error=valid_error)
    
    """
    def __init__(self, loss_key='loss', max_epochs=float('inf'), patience=20):
        self.loss_key = loss_key
        self.best_loss = float('inf')
        self.best_epoch = None

        self.max_epochs = max_epochs
        self.epoch = 0

        self.patience = patience
        self.frustration = 0
        self.improved = False
        self.converged = False

        self.monitors = defaultdict(list)
        self.start_time = time.time()

    def __getitem__(self, key):
        return self.monitors[key]

    def update(self, **kwargs):
        timestamp = time.time()

        for key, value in kwargs.items():
            self.monitors[key].append([timestamp, self.epoch, value])

        try:
            curr_loss = kwargs[self.loss_key]
            if curr_loss < self.best_loss:
                self.improved = True
                self.frustration = 0
                self.best_loss = curr_loss
                self.best_epoch = self.epoch
            else:
                self.improved = False
                self.frustration += 1
        except KeyError:
            pass

        self.epoch += 1
        self.converged = (
            self.patience < self.frustration or 
            self.epoch >= self.max_epochs)

    def numpy(self, key):
        n_fields = 3
        items = self.monitors[key]
        mat = np.zeros((n_fields, len(items)))
        for i in range(n_fields):
            mat[i] = [x[i] for x in items]
        return mat
