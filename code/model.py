import numpy as np
import sympy as sym
import scipy.optimize as opt

from sympy.utilities.lambdify import lambdify
from numpy import pi
from numpy import log, exp, sign, trace, dot
from numpy.linalg import inv, det

from snippets.stats import GP
from snippets.stats import circular_gaussian_kernel as make_kernel


def similarity(I0, I1, sf=1):
    """Computes the similarity between images I0 and I1"""
    # C = log(1. / sqrt(2 * pi * sf**2))
    diff = exp(np.sum(-0.5 * (I0 - I1)**2 / (sf**2))) + 1
    return diff


class GP_MarginalLogLikelihood(object):

    def __init__(self):
        d = sym.Symbol('d')
        h = sym.Symbol('h')
        w = sym.Symbol('w')

        k0 = (h**2 / (sym.sqrt(2 * sym.pi) * w))
        k1 = sym.exp(-(d)**2 / (2 * w**2))
        self.sym_K = k0 * k1

        self.sym_dK_dh = sym.diff(self.sym_K, h)
        self.dK_dh = lambdify(((h, w), d), self.sym_dK_dh)

        self.sym_dK_dw = sym.diff(self.sym_K, w)
        self.dK_dw = lambdify(((h, w), d), self.sym_dK_dw)

    @staticmethod
    def _d_dthetaj(Ki, dK_dthetaj, theta, x, y):
        dj = np.empty((x.size, x.size))
        for i in xrange(x.size):
            for j in xrange(x.size):
                d = x[i] - x[j]
                if abs(d) > pi:
                    diff = d - (sign(d) * 2 * pi)
                else:
                    diff = d
                dj[i, j] = dK_dthetaj(theta, diff)

        t0 = 0.5 * dot(y.T, dot(dot(Ki, dot(dj, Ki)), y))
        t1 = -0.5 * trace(dot(Ki, dj))
        dd = t0 + t1

        return dd

    def __call__(self, theta, x, y):
        """

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        (h, w) = theta

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = make_kernel(h, w, jit=False)(x, x)

        try:
            Ki = inv(K)
        except np.linalg.LinAlgError as err:
            print theta
            raise err

        t1 = -0.5 * dot(dot(y.T, Ki), y)
        t2 = -0.5 * log(det(K))
        t3 = -0.5 * x.size * log(2 * pi)
        mll = (t1 + t2 + t3)

        return -mll

    def jacobian(self, theta, x, y):
        h, w = theta
        K = make_kernel(h, w, jit=False)(x, x)

        try:
            Ki = inv(K)
        except np.linalg.LinAlgError as err:
            print theta
            raise err

        dmll = np.array([
            self._d_dthetaj(Ki, self.dK_dh, theta, x, y),
            self._d_dthetaj(Ki, self.dK_dw, theta, x, y),
        ])

        return -dmll

    def maximize(self, x, y):
        popt = opt.minimize(
            fun=self,
            x0=(1, 1),
            args=(x, y),
            method='L-BFGS-B',
            bounds=((1e-6, None), (1e-6, 2*pi)),
            jac=self.jacobian,
        )

        if not popt['success']:
            print popt
            raise RuntimeError("Could not estimate parameters")

        return tuple(popt['x'])


def fit_likelihood(ll, R, Sr, iix, cix):
    """

    References
    ----------

    Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
        Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
        Model Evidence Using Bayesian Quadrature. *Advances in Neural
        Information Processing Systems*, 25.

    """

    # input points for the GPs over S and log S
    xi = R[iix].copy()
    yi = Sr[iix].copy()

    # GPR for S
    print "Fitting parameters for GP over S..."
    theta_S = ll.maximize(xi, yi)
    print theta_S
    print "Computing GP over S..."
    mu_S, cov_S = GP(make_kernel(*theta_S), xi, yi, R)
    # GPR for log S
    print "Fitting parameters for GP over log S..."
    theta_logS = ll.maximize(xi, log(yi))
    print theta_logS
    print "Computing GP over log S..."
    mu_logS, cov_logS = GP(make_kernel(*theta_logS), xi, log(yi), R)

    # "candidate" points
    delta = mu_logS - log(mu_S)
    xc = R[cix].copy()
    yc = delta[cix].copy()

    # GPR for Delta
    print "Fitting parameters for GP over Dc..."
    theta_Dc = ll.maximize(xc, yc)
    print theta_Dc
    print "Computing GP over Dc..."
    mu_Dc, cov_Dc = GP(make_kernel(*theta_Dc), xc, yc, R)

    return (delta, xi, yi, xc, yc,
            mu_S, cov_S, mu_logS, cov_logS, mu_Dc, cov_Dc)
