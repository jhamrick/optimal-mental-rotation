import numpy as np
import sympy as sym
import scipy.optimize as opt

from sympy.utilities.lambdify import lambdify
from sympy.functions.special.delta_functions import DiracDelta as delta
from numpy import pi
from numpy import log, exp, sign, trace, dot, abs, diag
from numpy.linalg import inv

from snippets.stats import GP
from snippets.stats import periodic_kernel
from snippets.stats import gaussian_kernel


class KernelMLL(object):
    """Object representing the marginal log likelihood (MLL) of a
    kernel function.

    Methods
    -------
    __call__
        Compute the MLL.
    jacobian
        Compute the Jacobian of the MLL.
    maximize
        Find kernel parameter values which maximize the MLL.

    """

    def __init__(self, kernel, obs_noise=True):
        self.obs_noise = obs_noise

        # create symbolic variables
        d = sym.Symbol('d')
        h = sym.Symbol('h')
        w = sym.Symbol('w')
        s = sym.Symbol('s')

        # symbolic version of the kernel function
        if kernel == 'periodic':
            self.kernel = periodic_kernel
            k1 = (h ** 2) * sym.exp(-2. * (sym.sin(d / 2.) ** 2) / (w ** 2))
        elif kernel == 'gaussian':
            self.kernel = gaussian_kernel
            k1 = (h ** 2) * sym.exp(-0.5 * (d ** 2) / (w ** 2))
        else:
            raise ValueError("unsupported kernel '%s'" % kernel)

        if self.obs_noise:
            k2 = (s ** 2) * delta(d)
            self.sym_K = k1 + k2
        else:
            self.sym_K = k1

        # compute partial derivatives
        self.sym_dK_dh = sym.diff(self.sym_K, h)
        self.sym_dK_dw = sym.diff(self.sym_K, w)
        if self.obs_noise:
            self.sym_dK_ds = sym.diff(self.sym_K, s)

        # turn partial derivatives into functions, so we can evaluate
        # them (to compute the Jacobian)
        if self.obs_noise:
            self.dK_dh = lambdify(((h, w, s), d), self.sym_dK_dh)
            self.dK_dw = lambdify(((h, w, s), d), self.sym_dK_dw)
            self.dK_ds = lambdify(((h, w, s), d), self.sym_dK_ds)
        else:
            self.dK_dh = lambdify(((h, w), d), self.sym_dK_dh)
            self.dK_dw = lambdify(((h, w), d), self.sym_dK_dw)

    @staticmethod
    def _d_dthetaj(Ki, dK_dthetaj, theta, x, y):
        """Compute the partial derivative of the marginal log likelihood with
        respect to one of its parameters, `\theta_j`.

        This is computing Eq. 5.9 of Rasmussen & Williams (2006):

        $$\frac{\partial}{\partial\theta_j}\log{p(\mathbf{y} | X, \mathbf{\theta})}=\frac{1}{2}\mathbf{y}^\top K^{-1}\frac{\partial K}{\partial\theta_j}K^{-1}\mathbf{y}-frac{1}{2}\mathrm{tr}(K^{-1}\frac{\partial K}{\partial\theta_j})$$

        Parameters
        ----------
        Ki : numpy.ndarray with shape (n, n)
            Inverse of the kernel matrix `K`
        dK_dthetaj : function((h, w, s), d)
            Computes the partial derivative of `K` with respect to
            `theta_j`, where `theta_j` is `h`, `w`, or `s`. Takes as
            arguments the kernel parameters and the difference
            `x_1-x_2`.
        theta : 3-tuple
            The kernel parameters `h`, `w`, and `s`.
        x : numpy.ndarray with shape (n,)
            Given input values
        y : numpy.ndarray with shape (n,)
            Given output values

        Returns
        -------
        out : float
            Partial derivative of the marginal log likelihood.

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        # compute dK/dtheta_j matrix
        dj = np.empty((x.size, x.size))
        for i in xrange(x.size):
            for j in xrange(x.size):
                diff = x[i] - x[j]
                dj[i, j] = dK_dthetaj(theta, diff)

        # compute the partial derivative of the marginal log likelihood
        k = dot(Ki, dj)
        t0 = 0.5 * dot(y.T, dot(k, dot(Ki, y)))
        t1 = -0.5 * trace(k)
        dd = t0 + t1

        return dd

    def __call__(self, theta, x, y):
        """Computes the marginal negative log likelihood of the kernel.

        This is computing Eq. 5.8 of Rasmussen & Williams (2006):

        $$\log{p(\mathbf{y} | X,\mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \frac{1}{2}\log{|K_y|} - \frac{n}{2}\log{2\pi}

        Parameters
        ----------
        theta : 3-tuple
            The kernel parameters `h`, `w`, and `s`
        x : numpy.ndarray with shape (n,)
            The given input values
        y : numpy.ndarray with shape (n,)
            The given output values

        Returns
        -------
        out : float
            The marginal negative log likelihood of the data given the
            parameters

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        if self.obs_noise:
            h, w, s = theta
        else:
            h, w = theta
            s = 0

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = self.kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K and compute determinant
        L = np.linalg.cholesky(K)
        Li = inv(L)
        Ki = dot(Li.T, Li)
        logdetK = 2 * log(diag(L)).sum()

        # compute the marginal log likelihood
        t1 = -0.5 * dot(dot(y.T, Ki), y)
        t2 = -0.5 * logdetK
        t3 = -0.5 * x.size * log(2 * pi)
        mll = (t1 + t2 + t3)
        out = -mll

        return out

    def jacobian(self, theta, x, y):
        """Computes the negative Jacobian of the marginal log likelihood of
        the kernel.

        See Eq. 5.9 of Rasmussen & Williams (2006).

        Parameters
        ----------
        theta : 3-tuple
            The kernel parameters `h`, `w`, and `s`
        x : numpy.ndarray with shape (n,)
            The given input values
        y : numpy.ndarray with shape (n,)
            The given output values

        Returns
        -------
        out : 3-tuple
            The negative Jacobian of the marginal negative log
            likelihood of the data given the parameters

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        if self.obs_noise:
            h, w, s = theta
        else:
            h, w = theta
            s = 0

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = self.kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K
        Li = inv(np.linalg.cholesky(K))
        Ki = dot(Li.T, Li)

        dmll = [
            self._d_dthetaj(Ki, self.dK_dh, theta, x, y),
            self._d_dthetaj(Ki, self.dK_dw, theta, x, y)
        ]

        if self.obs_noise:
            dmll.append(self._d_dthetaj(Ki, self.dK_ds, theta, x, y))

        return -np.array(dmll)

    def maximize(self, x, y,
                 hmin=1e-8, wmin=1e-8,
                 ntry=10, verbose=False):
        """Find kernel parameter values which maximize the marginal log
        likelihood of the data.

        Parameters
        ----------
        x : numpy.ndarray with shape (n,)
            The given input values
        y : numpy.ndarray with shape (n,)
            The given output values
        ntry : integer (default=10)
           Number of times to run MLII, to try to avoid local maxima
        verbose : bool (default=False)
           Print information about the optimization

        Returns
        -------
        out : 3-tuple
            The parameter values corresponding to the maximum marginal
            log likelihood (that was found -- there might be a better
            set of parameters, as this only finds local optima).

        """

        if self.obs_noise:
            args = np.empty((ntry, 3))
        else:
            args = np.empty((ntry, 2))
        fval = np.empty(ntry)

        for i in xrange(ntry):
            # randomize starting parameter values
            h0 = np.random.uniform(0, np.ptp(y)**2)
            w0 = np.random.uniform(0, 2*np.pi)
            if self.obs_noise:
                s0 = np.random.uniform(0, np.sqrt(np.var(y)))
                p0 = (h0, w0, s0)
                method = "BFGS"
                bounds = None
            else:
                p0 = (h0, w0)
                method = "L-BFGS-B"
                bounds = ((hmin, None), (wmin, None))

            # run mimization function
            try:
                popt = opt.minimize(
                    fun=self,
                    x0=p0,
                    args=(x, y),
                    jac=self.jacobian,
                    method=method,
                    bounds=bounds
                )
            except np.linalg.LinAlgError as e:
                # kernel matrix is not positive definite, probably
                success = False
                message = str(e)
            else:
                # get results of the optimization
                success = popt['success']
                message = popt['message']

            if not success:
                args[i] = np.nan
                fval[i] = np.inf
                if verbose:
                    print "Failed: %s" % message
            else:
                args[i] = abs(popt['x'])
                fval[i] = popt['fun']
                if verbose:
                    print "-MLL(%s) = %f" % (args[i], fval[i])

        # choose the parameters that give the best MLL
        best = np.argmin(fval)
        if np.isinf(fval[best]) and sign(fval[best]) > 0:
            print args[best], fval[best]
            raise RuntimeError("Could not find MLII parameter estimates")

        return args[best]


def similarity(I0, I1, sf=1):
    """Computes the similarity between images `I0` and `I1`."""
    S = exp(np.sum(-0.5 * ((I0 - I1) ** 2) / (sf ** 2)))
    return S


class BayesianQuadrature(object):
    """Estimate a likelihood function, S(y|x) using Gaussian Process
    regressions, as in Osborne et al. (2012):

    1) Estimate S using a GP
    2) Estimate log(S) using second GP
    3) Estimate delta_C using a third GP

    References
    ----------
    Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
        Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
        Model Evidence Using Bayesian Quadrature. *Advances in Neural
        Information Processing Systems*, 25.

    """

    def __init__(self, x, y, mll, ntry=10, verbose=False):
        """Initialize the likelihood estimator object.

        Parameters
        ----------
        mll : KernelMLL object
        x : numpy.ndarray
            Vector of x values
        y : numpy.ndarray
            Vector of (actual) y values
        verbose : bool (default=False)
            Whether to print information during fitting

        """

        # true x and y values for the likelihood
        self.x = x.copy()
        self.y = y.copy()
        # marginal log likelihood object
        self.mll = mll
        # number of ML tries
        self.ntry = ntry
        # print fitting information
        self.verbose = verbose

    def _fit_gp(self, xi, yi, name, wmin=1e-8):
        if self.verbose:
            print "Fitting parameters for GP over %s ..." % name

        # fit parameters
        theta = self.mll.maximize(
            xi, yi, wmin=wmin, ntry=self.ntry, verbose=self.verbose)

        if self.verbose:
            print theta
            print "Computing GP over %s..." % name

        # GP regression
        mu, cov = GP(
            self.mll.kernel(*theta), xi, yi, self.x)

        return mu, cov, theta

    def fit(self, iix):
        """Run the GP regressions to fit the likelihood function.

        Parameters
        ----------
        iix : numpy.ndarray
            Integer array of indices corresponding to the "given" x and y data
        cix : numpy.ndarray
            Integer array of indices corresponding to the "candidate" x points.

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        # input data
        self.xi = self.x[iix].copy()
        self.yi = self.y[iix].copy()

        # compute GP regressions for S and log(S)
        self.mu_S, self.cov_S, self.theta_S = self._fit_gp(
            self.xi, self.yi, "S")
        self.mu_logS, self.cov_logS, self.theta_logS = self._fit_gp(
            self.xi, log(self.yi + 1), "log(S)")

        # choose "candidate" points, halfway between given points
        cix = cix = np.sort(np.unique(np.concatenate([
            (iix + np.array(list(iix[1:]) + [self.x.size])) / 2,
            iix])))
        self.delta = self.mu_logS - log(self.mu_S + 1)
        self.xc = self.x[cix].copy()
        self.yc = self.delta[cix].copy()

        # compute GP regression for Delta_c
        self.mu_Dc, self.cov_Dc, self.theta_Dc = self._fit_gp(
            self.xc, self.yc, "Delta_c",
            wmin=min(self.theta_S[1], self.theta_logS[1]) / 2.)

        # mean of the final regression for S
        self.mean = ((self.mu_S + 1) * (1 + self.delta)) - 1


# def dm_dw(mll, theta, x, y, xo):
#     h, w, s = theta

#     # the overhead of JIT compiling isn't it worth it here because
#     # this is just a temporary kernel function
#     K = kernel(h, w, jit=False)(x, x)
#     if s > 0:
#         K += np.eye(x.size) * (s ** 2)

#     # invert K
#     Li = inv(np.linalg.cholesky(K))
#     Ki = dot(Li.T, Li)

#     # get the partial derivative function
#     dK_dw = mll.dK_dw

#     # compute dK/dtheta_j matrix
#     dKxx = np.empty((x.size, x.size))
#     for i in xrange(x.size):
#         for j in xrange(x.size):
#             diff = x[i] - x[j]
#             dKxx[i, j] = dK_dw((h, w, s), diff)

#     dKxxo = np.empty((x.size, xo.size))
#     for i in xrange(x.size):
#         for j in xrange(xo.size):
#             diff = x[i] - xo[j]
#             dKxxo[i, j] = dK_dw((h, w, s), diff)

#     dKi = dot(-Ki, dot(dKxx, Ki))
#     dm = dot(dKxxo.T, dot(dKi, y))
#     return dm
