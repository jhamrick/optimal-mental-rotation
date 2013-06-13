import numpy as np
from numpy import dot
np.seterr(all='raise')

from models.kernels import GaussianKernel as kernel
from models.gaussian_process import GP
from snippets.safemath import EPS
from util import load_opt, rand_params

######################################################################


class TestKernels(object):

    def __init__(self):
        opt = load_opt()
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = np.sqrt(EPS) * 10

    def check_gp_mean(self, gp, y):
        diff = np.abs(gp.m - y)
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad gp mean")

    def check_gp_inv(self, gp):
        I = dot(gp.Kxx, gp.inv_Kxx)
        diff = np.abs(I - np.eye(I.shape[0]))
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad inverted kernel matrix")

    def test_gp_mean(self):
        x = xo = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 's')
            gp = GP(kernel(*params), x, y, xo)
            yield self.check_gp_mean, gp, y

    def test_gp_inv(self):
        x = xo = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 's')
            gp = GP(kernel(*params), x, y, xo)
            yield self.check_gp_inv, gp

    # def check_L(mat0):
    #     L = cholesky(mat0)
    #     mat = np.dot(L, L.T)
    #     diff = np.abs(mat0 - mat)
    #     if not (diff < EPS).all():
    #         print mat0
    #         print mat
    #         assert False

    # def test_cholesky(self):
    #     n = 10
    #     for i in xrange(self.N_big):
    #         x = np.random.rand(n) * 10
    #         d = x[:, None] - x[None, :]
    #         mat0 = scipy.stats.norm.pdf(d)
    #         yield check_L, mat0
