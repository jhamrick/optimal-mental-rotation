import scipy.stats
import numpy as np
np.seterr(all='raise')

from models.kernels import GaussianKernel, PeriodicKernel
from snippets.safemath import EPS
from util import load_opt

######################################################################


def rand_params(*args):
    params = []
    for param in args:
        if param == 'h':
            params.append(np.random.uniform(0, 2))
        elif param == 'w':
            params.append(np.random.uniform(np.pi / 32., np.pi / 2.))
        elif param == 'p':
            params.append(np.random.uniform(0.33, 3))
        elif param == 's':
            params.append(np.random.uniform(0, 1))
    return tuple(params)


def approx_deriv(y0, y1, dx):
    dy = (y1 - y0) / 2.
    return dy / dx


class TestKernels(object):

    def __init__(self):
        opt = load_opt()
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = np.sqrt(EPS)

    def check_params(self, kernel, params):
        k = kernel(*params)
        if k.params != params:
            print k.params
            print params
            raise AssertionError("parameters do not match")

    def check_gaussian_K(self, x, dx, params):
        kernel = GaussianKernel(*params)
        K = kernel(x, x)
        print "Kernel parameters:", kernel.params

        h, w, s = params
        pdx = scipy.stats.norm.pdf(dx, loc=0, scale=w)
        pdx *= (h ** 2) * np.sqrt(2 * np.pi) * w
        pdx[dx == 0] += s ** 2

        diff = abs(pdx - K)
        if not (diff < self.thresh).all():
            print self.thresh, diff
            raise AssertionError("invalid gaussian kernel matrix")

    def check_periodic_K(self, x, dx, params):
        kernel = PeriodicKernel(*params)
        K = kernel(x, x)
        print "Kernel parameters:", kernel.params

        h, w, p, s = params
        pdx = (h ** 2) * np.exp(-2. * (np.sin(dx / (2. * p)) ** 2) / (w ** 2))
        pdx[dx == 0] += s ** 2

        diff = abs(pdx - K)
        if not (diff < self.thresh).all():
            print self.thresh, diff
            raise AssertionError("invalid periodic kernel matrix")

    def check_jacobian(self, kernel, params, x):
        k = kernel(*params)
        jac = k.jacobian(x, x)
        dtheta = self.thresh * 10

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= dtheta
            p1 = list(params)
            p1[i] += dtheta
            k0 = kernel(*p0)(x, x)
            k1 = kernel(*p1)(x, x)
            approx_jac[i] = approx_deriv(k0, k1, dtheta)

        diff = jac - approx_jac
        if (np.abs(diff) > self.thresh).any():
            print self.thresh, diff
            raise AssertionError("bad jacobian")

    def check_hessian(self, kernel, params, x):
        k = kernel(*params)
        hess = k.hessian(x, x)
        dtheta = self.thresh * 10

        approx_hess = np.empty(hess.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p1 = list(params)
            p0[i] -= dtheta
            p1[i] += dtheta
            jac0 = kernel(*p0).jacobian(x, x)
            jac1 = kernel(*p1).jacobian(x, x)
            approx_hess[:, i] = approx_deriv(jac0, jac1, dtheta)

        diff = hess - approx_hess
        if (np.abs(diff) > self.thresh).any():
            print self.thresh, np.abs(diff).max()
            raise AssertionError("bad hessian")

    ######################################################################

    def test_gaussian_kernel_params(self):
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 's')
            yield self.check_params, GaussianKernel, params

    def test_periodic_kernel_params(self):
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 'p', 's')
            yield self.check_params, PeriodicKernel, params

    def test_gaussian_K(self):
        """Test stats.gaussian_kernel output matrix"""
        x = np.linspace(-2, 2, 10)
        dx = x[:, None] - x[None, :]
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 's')
            yield (self.check_gaussian_K, x, dx, params)

    def test_periodic_K(self):
        """Test stats.periodic_kernel output matrix"""
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        dx = x[:, None] - x[None, :]
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 'p', 's')
            yield (self.check_periodic_K, x, dx, params)

    def test_gaussian_jacobian(self):
        x = np.linspace(-2, 2, 10)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 's')
            yield (self.check_jacobian, GaussianKernel, params, x)

    def test_gaussian_hessian(self):
        x = np.linspace(-2, 2, 10)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 's')
            yield (self.check_hessian, GaussianKernel, params, x)
