import unittest
import numpy as np
import torch
from easynet.functions import attend


def FloatVar(x):
    return torch.autograd.Variable(torch.FloatTensor(x))


class AttentionTest(unittest.TestCase):
    n_mem = 3
    v_dim = 4
    k_dim = 5
    batch_size = 2

    Q = [[ 5,  4, -1,  2],
         [ 5,  4,  2,  1]]

    V = [[[ 5, -5, -1, -2],
          [ 1,  0,  5,  0],
          [ 1, -3, -4,  1]],
         [[-1,  3,  5,  0],
          [-1, -4,  4,  1],
          [ 0,  0, -1,  4]]]

    K = [[[ 5,  2, -1,  0, -5],
          [ 1,  4, -1, -5, -5],
          [-4, -5,  0, -3,  3]],
         [[ 3,  5,  1, -3, -3],
          [-4, -2,  4,  3, -5],
          [ 2, -4, -1,  0,  4]]]

    p_q0_V0 = [0.8437947344813395, 0.11419519938459449, 0.04201006613406605]
    sum_p_q0_V0_V0 = [4.375178937925358, -4.345003870808895, -0.44085900209463125, -1.6455794028286128]
    sum_p_q0_V0_K0 = [4.165128607255028, 1.9343199358307268, -0.9579899338659339, -0.6970061953251705, -4.663919470927472]

    p_q1_V0 = [8.315280276639204e-07, 0.9999991684717178, 2.543663532246996e-13]
    sum_p_q1_V0_V0 = [1.0000033261121106, -4.157640901418662e-06, 4.999995010829543, -1.6630558009614876e-06]
    sum_p_q1_V0_K0 = [1.0000033261108388, 3.9999983369416547, -0.9999999999997454, -4.999995842359351, -4.999999999997963]

    p_q1_V1 = [0.9999996940975185, 2.5436648692632894e-13, 3.0590222692554685e-07]
    sum_p_q1_V1_V1 = [-0.9999996940977729, 2.9999990822915383, 4.999998164586383, 1.2236091620686743e-06]
    sum_p_q1_V1_K1 = [2.999999694095992, 4.999997246878176, 0.9999993881963092, -2.999999082291793, -2.99999785868492]
    
    def test_1xn_1xn(self):
        q = FloatVar([self.Q[0]])
        vs = [FloatVar([v]) for v in self.V[0]]
        ks = [FloatVar([k]) for k in self.K[0]]
        
        p = np.hstack(p.data.numpy() for p in attend(q, vs))
        self.assertEqual(p.shape, (1, self.n_mem))
        self.assertTrue(np.allclose(p, np.array([self.p_q0_V0])))

        h = attend(q, vs, vs).data.numpy()
        self.assertEqual(h.shape, (1, self.v_dim))
        self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_V0_V0])))

        g = attend(q, vs, ks).data.numpy()
        self.assertEqual(g.shape, (1, self.k_dim))
        self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_V0_K0])))
        
    def test_mxn_1xn(self):
        q = FloatVar(self.Q)
        vs = [FloatVar([v]).expand(self.batch_size, self.v_dim) for v in self.V[0]]
        ks = [FloatVar([k]).expand(self.batch_size, self.k_dim) for k in self.K[0]]
        
        p = np.hstack(p.data.numpy() for p in attend(q, vs))
        self.assertEqual(p.shape, (self.batch_size, self.n_mem))
        self.assertTrue(np.allclose(p, np.array([self.p_q0_V0, self.p_q1_V0])))

        h = attend(q, vs, vs).data.numpy()
        self.assertEqual(h.shape, (self.batch_size, self.v_dim))
        self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_V0_V0, self.sum_p_q1_V0_V0])))

        g = attend(q, vs, ks).data.numpy()
        self.assertEqual(g.shape, (self.batch_size, self.k_dim))
        self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_V0_K0, self.sum_p_q1_V0_K0])))

    def test_mxn_mxn(self):
        q = FloatVar(self.Q)
        vs = [FloatVar(v) for v in zip(*self.V)]
        ks = [FloatVar(k) for k in zip(*self.K)]
        
        p = np.hstack(p.data.numpy() for p in attend(q, vs))
        self.assertEqual(p.shape, (self.batch_size, self.n_mem))
        self.assertTrue(np.allclose(p, np.array([self.p_q0_V0, self.p_q1_V1])))

        h = attend(q, vs, vs).data.numpy()
        self.assertEqual(h.shape, (self.batch_size, self.v_dim))
        self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_V0_V0, self.sum_p_q1_V1_V1])))

        g = attend(q, vs, ks).data.numpy()
        self.assertEqual(g.shape, (self.batch_size, self.k_dim))
        self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_V0_K0, self.sum_p_q1_V1_K1])))

    def test_things_that_should_fail(self):
        with self.assertRaises(RuntimeError):
            # ks is empty
            q = FloatVar(self.Q)
            attend(q, [])

        with self.assertRaises(RuntimeError):
            # vs is empty
            q = FloatVar(self.Q)
            ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
            attend(q, ks, [])

        with self.assertRaises(RuntimeError):
            # len(ks) != len(vs)
            q = FloatVar(self.Q)
            ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
            vs = [FloatVar(np.random.random((self.batch_size, self.v_dim)))]
            attend(q, ks, vs)

        with self.assertRaises(RuntimeError):
            # ks[i] has wrong batch_size
            q = FloatVar(self.Q)
            ks = [FloatVar(np.random.random((1, self.k_dim))) for i in range(2)]
            attend(q, ks)

        with self.assertRaises(RuntimeError):
            # ks[i] has wrong batch_size
            q = FloatVar(self.Q)
            ks = [FloatVar(np.random.random((self.batch_size + 1, self.k_dim))) for i in range(2)]
            attend(q, ks)

        with self.assertRaises(RuntimeError):
            # vs[i] has wrong batch_size
            q = FloatVar(self.Q)
            ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
            vs = [FloatVar(np.random.random((self.batch_size + 1, self.v_dim))) for i in range(2)]
            attend(q, ks, vs)


if __name__ == '__main__':
    unittest.main()
