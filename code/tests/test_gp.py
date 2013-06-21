import numpy as np
from numpy import dot
np.seterr(all='raise')

from models.kernels import GaussianKernel as kernel
from models.gaussian_process import GP
from snippets.safemath import EPS
from util import load_opt, rand_params, approx_deriv

######################################################################


class TestGP(object):

    def __init__(self):
        opt = load_opt()
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = 1e-6
        self.dtheta = np.sqrt(EPS) * 100

    def check_mean(self, gp, y):
        diff = np.abs(gp.mean(gp.x) - y)
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad gp mean")

    def check_inv(self, gp):
        I = dot(gp.Kxx, gp.inv_Kxx)
        diff = np.abs(I - np.eye(I.shape[0]))
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad inverted kernel matrix")

    def check_dloglh(self, gp, params):
        jac = gp.dloglh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.log_lh, gp1.log_lh, self.dtheta)

        diff = np.abs(jac - approx_jac)
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print jac
            print approx_jac
            raise AssertionError("bad dloglh_dtheta")

    def check_dlh(self, gp, params):
        jac = gp.dlh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.lh, gp1.lh, self.dtheta)

        diff = jac - approx_jac
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print jac
            print approx_jac
            raise AssertionError("bad dlh_dtheta")

    def check_d2lh(self, gp, params):
        hess = gp.d2lh_dtheta2

        approx_hess = np.empty(hess.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_hess[:, i] = approx_deriv(
                gp0.dlh_dtheta, gp1.dlh_dtheta, self.dtheta)

        diff = hess - approx_hess
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print hess
            print approx_hess
            raise AssertionError("bad d2lh_dtheta2")

    ##################################################################

    def test_mean(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = 0
            gp = GP(kernel(*params), x, y, s=s)
            yield self.check_mean, gp, y

    def test_inv(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            yield self.check_inv, gp

    def test_dloglh(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            yield self.check_dloglh, gp, params + (s,)

    def test_dlh(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            yield self.check_dlh, gp, params + (s,)

    def test_d2lh(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        y = np.sin(x)
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            yield self.check_d2lh, gp, params + (s,)
