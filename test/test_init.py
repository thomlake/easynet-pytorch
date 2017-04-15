from __future__ import absolute_import

import unittest
import numpy as np
import torch
from torch import FloatTensor
from torch.nn import Parameter

from easynet import init


class InitTest(unittest.TestCase):
    def test_orthonormal(self):
        w = init.orthonormal(FloatTensor(5, 5), gain=1)
        m = w.numpy()
        s = np.linalg.svd(m)[1]
        self.assertTrue(np.allclose(s, 1))

    def test_sparse(self):
        shapes = [(1, 1), (2, 1), (1, 2), (10, 7), (7, 10)]
        for shape in shapes:
            w = init.sparse(FloatTensor(*shape).zero_())
            self.assertFalse(np.any(np.isclose(np.abs(w.numpy()).sum(1), 0)))
            self.assertFalse(np.any(np.isclose(np.abs(w.numpy()).sum(0), 0)))


if __name__ == '__main__':
    unittest.main()
