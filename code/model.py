import numpy as np
import sympy as sym
import scipy.optimize as opt

from sympy.utilities.lambdify import lambdify
from sympy.functions.special.delta_functions import DiracDelta as delta
from numpy import pi
from numpy import log, exp, sign, trace, dot, abs, diag
from numpy.linalg import inv

from snippets.stats import GP
from snippets.stats import periodic_kernel as periodic_kernel


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
        s = sym.Symbol('s')

        k1 = (h ** 2) * sym.exp(-2. * sym.sin(d / 2.) ** 2 / (w ** 2))
        k2 = (s ** 2) * delta(d)
        self.sym_K = k1 + k2

        self.sym_dK_dh = sym.diff(self.sym_K, h)
        self.dK_dh = lambdify(((h, w, s), d), self.sym_dK_dh)

        self.sym_dK_dw = sym.diff(self.sym_K, w)
        self.dK_dw = lambdify(((h, w, s), d), self.sym_dK_dw)

        self.sym_dK_ds = sym.diff(self.sym_K, s)
        self.dK_ds = lambdify(((h, w, s), d), self.sym_dK_ds)

    @staticmethod
    def _d_dthetaj(Ki, dK_dthetaj, theta, x, y):
        h, w, s = theta
        dj = np.empty((x.size, x.size))
        for i in xrange(x.size):
            for j in xrange(x.size):
                diff = x[i] - x[j]
                dj[i, j] = dK_dthetaj((h, w, s), diff)

        k = dot(Ki, dj)
        t0 = 0.5 * dot(y.T, dot(k, dot(Ki, y)))
        t1 = -0.5 * trace(k)
        dd = t0 + t1

        return dd

    def __call__(self, theta, x, y):
        """

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        h, w, s = theta

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = periodic_kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K
        L = np.linalg.cholesky(K)
        Li = inv(L)
        Ki = dot(Li.T, Li)

        t1 = -0.5 * dot(dot(y.T, Ki), y)
        # The original computation here was -0.5 * log(det(K)), but
        # this equals -0.5 * log(prod(diag(L))**2).
        t2 = -log(diag(L)).sum()
        t3 = -0.5 * x.size * log(2 * pi)
        mll = (t1 + t2 + t3)
        out = -mll

        return out

    def jacobian(self, theta, x, y):

        h, w, s = theta

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = periodic_kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K
        Li = inv(np.linalg.cholesky(K))
        Ki = dot(Li.T, Li)

        dmll = np.array([
            self._d_dthetaj(Ki, self.dK_dh, theta, x, y),
            self._d_dthetaj(Ki, self.dK_dw, theta, x, y),
            self._d_dthetaj(Ki, self.dK_ds, theta, x, y),
        ])

        return -dmll

    def maximize(self, x, y, ntry=10):
        args = np.empty((ntry, 3))
        fval = np.empty(ntry)

        for i in xrange(ntry):
            h0 = np.random.uniform(0, np.ptp(y)**2)
            w0 = np.random.uniform(0, np.pi)
            s0 = np.random.uniform(0, np.sqrt(np.var(y)))

            try:
                popt = opt.minimize(
                    fun=self,
                    x0=(h0, w0, s0),
                    args=(x, y),
                    # method='L-BFGS-B',
                    # tol=1e-4,
                    # bounds=((1e-6, None), (1e-6, pi), (0, None)),
                    jac=self.jacobian,
                )
            except ArithmeticError:
                success = False
            except np.linalg.LinAlgError:
                success = False
            else:
                success = popt['success']

            if not success:
                args[i] = np.nan
                fval[i] = np.inf
            else:
                args[i] = abs(popt['x'])
                fval[i] = popt['fun']

                print "-MLL(%s) = %f" % (
                    args[i], fval[i])

        best = np.argmin(fval)
        if np.isinf(fval[best]) and sign(fval[best]) > 0:
            print args[best], fval[best]
            raise RuntimeError("Could not find MLII parameter estimates")

        return args[best]


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
    mu_S, cov_S = GP(periodic_kernel(*theta_S), xi, yi, R)
    # GPR for log S
    print "Fitting parameters for GP over log S..."
    theta_logS = ll.maximize(xi, log(yi))
    print theta_logS
    print "Computing GP over log S..."
    mu_logS, cov_logS = GP(periodic_kernel(*theta_logS), xi, log(yi), R)

    # "candidate" points
    delta = mu_logS - log(mu_S)
    xc = R[cix].copy()
    yc = delta[cix].copy()

    # GPR for Delta
    print "Fitting parameters for GP over Dc..."
    theta_Dc = ll.maximize(xc, yc)
    print theta_Dc
    print "Computing GP over Dc..."
    mu_Dc, cov_Dc = GP(periodic_kernel(*theta_Dc), xc, yc, R)

    return (delta, xi, yi, xc, yc,
            mu_S, cov_S, mu_logS, cov_logS, mu_Dc, cov_Dc)
