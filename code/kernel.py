import numpy as np
import scipy.optimize as opt
import sympy as sym

from sympy.utilities.lambdify import lambdify
from sympy.functions.special.delta_functions import DiracDelta as delta

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

    def __init__(self, kernel, **params):

        # create symbolic variables
        d = sym.Symbol('d')

        free_params = []
        # output scale (height)
        if params.get('h', None):
            h = params['h']
        else:
            h = sym.Symbol('h')
            free_params.append(h)
        # input scale (width)
        if params.get('w', None):
            w = params['w']
        else:
            w = sym.Symbol('w')
            free_params.append(w)
        # observation noise
        if params.get('s', None) is not None:
            s = params['s']
        else:
            s = sym.Symbol('s')
            free_params.append(s)

        # symbolic version of the kernel function
        if kernel == 'periodic':
            self.kernel = periodic_kernel
            k1 = (h ** 2) * sym.exp(-2. * (sym.sin(d / 2.) ** 2) / (w ** 2))
        elif kernel == 'gaussian':
            self.kernel = gaussian_kernel
            k1 = (h ** 2) * sym.exp(-0.5 * (d ** 2) / (w ** 2))
        else:
            raise ValueError("unsupported kernel '%s'" % kernel)

        k2 = (s ** 2) * delta(d)
        self.sym_K = k1 + k2

        # compute partial derivatives adn turn them into functions, so
        # we can evaluate them (to compute the Jacobian)
        self.sym_dK_dtheta = []
        self.dK_dtheta = []
        for p in free_params:
            sym_f = sym.diff(self.sym_K, p)
            f = lambdify((tuple(free_params), d), sym_f)
            self.sym_dK_dtheta.append(sym_f)
            self.dK_dtheta.append(f)

        self.h = None if type(h) is sym.Symbol else h
        self.w = None if type(w) is sym.Symbol else w
        self.s = None if type(s) is sym.Symbol else s

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
        k = np.dot(Ki, dj)
        t0 = 0.5 * np.dot(y.T, np.dot(k, np.dot(Ki, y)))
        t1 = -0.5 * np.trace(k)
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

        th = list(theta)
        h = th.pop(0) if self.h is None else self.h
        w = th.pop(0) if self.w is None else self.w
        s = th.pop(0) if self.s is None else self.s

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = self.kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K and compute determinant
        L = np.linalg.cholesky(K)
        Li = np.linalg.inv(L)
        Ki = np.dot(Li.T, Li)
        logdetK = 2 * np.log(np.diag(L)).sum()

        # compute the marginal log likelihood
        t1 = -0.5 * np.dot(np.dot(y.T, Ki), y)
        t2 = -0.5 * logdetK
        t3 = -0.5 * x.size * np.log(2 * np.pi)
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

        th = list(theta)
        h = th.pop(0) if self.h is None else self.h
        w = th.pop(0) if self.w is None else self.w
        s = th.pop(0) if self.s is None else self.s

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = self.kernel(h, w, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)

        # invert K
        Li = np.linalg.inv(np.linalg.cholesky(K))
        Ki = np.dot(Li.T, Li)

        dmll = [self._d_dthetaj(Ki, dK_dthetaj, theta, x, y)
                for dK_dthetaj in self.dK_dtheta]

        return -np.array(dmll)

    def maximize(self, x, y,
                 hmin=1e-8, hmax=None,
                 wmin=1e-8, wmax=None,
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

        args = np.empty((ntry, len(self.dK_dtheta)))
        fval = np.empty(ntry)

        for i in xrange(ntry):
            # randomize starting parameter values
            p0 = []
            bounds = []
            if self.h is None:
                p0.append(np.random.uniform(0, np.max(np.abs(y))*2))
                bounds.append((hmin, hmax))
            if self.w is None:
                p0.append(np.random.uniform(0, 2*np.pi))
                bounds.append((wmin, wmax))
            if self.s is None:
                p0.append(np.random.uniform(0, np.sqrt(np.var(y))))
                bounds.append((0, None))

            p0 = tuple(p0)
            bounds = tuple(bounds)
            method = "L-BFGS-B"

            if verbose:
                print "      p0 = %s" % (p0,)

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
                    print "      Failed: %s" % message
            else:
                args[i] = abs(popt['x'])
                fval[i] = popt['fun']
                if verbose:
                    print "      -MLL(%s) = %f" % (args[i], fval[i])

        # choose the parameters that give the best MLL
        best = np.argmin(fval)
        if np.isinf(fval[best]) and np.sign(fval[best]) > 0:
            print args[best], fval[best]
            raise RuntimeError("Could not find MLII parameter estimates")

        th = list(args[best])
        h = th.pop(0) if self.h is None else self.h
        w = th.pop(0) if self.w is None else self.w
        s = th.pop(0) if self.s is None else self.s

        return (h, w, s)
