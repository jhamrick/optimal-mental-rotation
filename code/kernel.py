import numpy as np
import scipy.optimize as opt
import sympy as sym

from sympy.utilities.lambdify import lambdify
from sympy.functions.special.delta_functions import DiracDelta as delta

from snippets.stats import periodic_kernel
from snippets.stats import gaussian_kernel
from snippets.stats import MIN_LOG, MAX_LOG


def cholesky(mat):
    m = np.mean(np.abs(mat))
    try:
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        # matrix is singular, let's try adding some noise and see if
        # we can invert it then
        noise = np.random.normal(0, m * 1e-4, mat.shape)
        try:
            L = np.linalg.cholesky(mat + noise)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "Could not compute Cholesky decomposition of "
                "kernel matrix, even with jitter")
    return L


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
            h = sym.Symbol('h', positive=True)
            free_params.append(h)
        # input scale (width)
        if params.get('w', None):
            w = params['w']
        else:
            w = sym.Symbol('w', positive=True)
            free_params.append(w)
        # observation noise
        if params.get('s', None) is not None:
            s = params['s']
        else:
            s = sym.Symbol('s', positive=True)
            free_params.append(s)
        # period
        if params.get('p', None) or kernel != 'periodic':
            p = params.get('p', 1)
        else:
            p = sym.Symbol('p', positive=True)
            free_params.append(p)

        # parameters for lambdify
        if len(free_params) == 0:
            lparams = (d,)
        else:
            lparams = (tuple(free_params), d)

        # symbolic version of the kernel function
        self.kernel_type = kernel
        if kernel == 'periodic':
            self.kernel = periodic_kernel
            k1 = (h ** 2) * sym.exp(-2.*(sym.sin(d / (2.*p)) ** 2) / (w ** 2))
        elif kernel == 'gaussian':
            self.kernel = gaussian_kernel
            k1 = (h ** 2) * sym.exp(-0.5 * (d ** 2) / (w ** 2))
        else:
            raise ValueError("unsupported kernel '%s'" % kernel)

        k2 = (s ** 2) * delta(d)
        self.sym_K = k1 + k2
        self.K = lambdify(lparams, self.sym_K)

        # compute 1st and 2nd partial derivatives adn turn them into
        # functions, so we can evaluate them (to compute the Jacobian
        # and Hessian)
        self.sym_dK_dtheta = []
        self.dK_dtheta = []
        self.sym_d2K_dtheta2 = [[]]*len(free_params)
        self.d2K_dtheta2 = [[]]*len(free_params)

        for i, p1 in enumerate(free_params):
            # first partial derivatives
            sym_f = sym.diff(self.sym_K, p1)
            f = lambdify(lparams, sym_f)
            self.sym_dK_dtheta.append(sym_f)
            self.dK_dtheta.append(f)

            for p2 in free_params:
                # second partial derivatives
                sym_df = sym.diff(sym_f, p2)
                df = lambdify(lparams, sym_df)
                self.sym_d2K_dtheta2[i].append(sym_df)
                self.d2K_dtheta2[i].append(df)

        self.h = None if type(h) is sym.Symbol else h
        self.w = None if type(w) is sym.Symbol else w
        self.s = None if type(s) is sym.Symbol else s
        self.p = None if type(p) is sym.Symbol else p

    def kernel_params(self, theta):
        th = list(theta)
        h = th.pop(0) if self.h is None else self.h
        w = th.pop(0) if self.w is None else self.w
        p = th.pop(0) if self.p is None else self.p
        s = th.pop(0) if self.s is None else self.s
        return h, w, p, s

    def make_kernel(self, theta=None, params=None, jit=True):
        if not params:
            h, w, p, s = self.kernel_params(theta)
        else:
            h, w, p, s = params

        if self.kernel_type == 'periodic':
            k = self.kernel(h, w, p, jit=jit)
        else:
            k = self.kernel(h, w, jit=jit)

        return k

    def Kxx(self, theta, x):
        params = self.kernel_params(theta)
        h, w, p, s = params

        # the overhead of JIT compiling isn't it worth it here because
        # this is just a temporary kernel function
        K = self.make_kernel(params=params, jit=False)(x, x)
        if s > 0:
            K += np.eye(x.size) * (s ** 2)
        return K

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
        dK_dthetaj : function(theta, d)
            Computes the partial derivative of `K` with respect to
            `theta_j`. Takes as arguments the kernel parameters and the
            difference `x_1-x_2`.
        theta : tuple
            The kernel parameters.
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
                try:
                    dj[i, j] = dK_dthetaj(theta, diff)
                except ArithmeticError:
                    dj[i, j] = np.nan

        # compute the partial derivative of the marginal log likelihood
        k = np.dot(Ki, dj)
        t0 = 0.5 * np.dot(y.T, np.dot(k, np.dot(Ki, y)))
        t1 = -0.5 * np.trace(k)
        dd = t0 + t1

        return dd

    @staticmethod
    def _d2_dthetajdthetai(Ki, dK_dthetaj, dK_dthetai, d2K_dthetaji,
                           theta, x, y):
        """Compute the second partial derivative of the marginal log
        likelihood with respect to two of its parameters, `\theta_j`
        and `\theta_i`.

        This is computing the derviative of Eq. 5.9 of Rasmussen &
        Williams (2006):

        $$\frac{\partial}{\partial\theta_j}\log{p(\mathbf{y} | X, \mathbf{\theta})}=\frac{1}{2}\mathbf{y}^\top K^{-1}\frac{\partial K}{\partial\theta_j}K^{-1}\mathbf{y}-frac{1}{2}\mathrm{tr}(K^{-1}\frac{\partial K}{\partial\theta_j})$$

        Parameters
        ----------
        Ki : numpy.ndarray with shape (n, n)
            Inverse of the kernel matrix `K`
        dK_dthetaj : function(theta, d)
            Computes the partial derivative of `K` with respect to
            `theta_j`.
        dK_dthetai : function(theta, d)
            Computes the partial derivative of `K` with respect to
            `theta_i`.
        dK_dthetaji : function(theta, d)
            Computes the second partial derivative of `K` with respect
            to `theta_j`, and `theta_i`.
        theta : tuple
            The kernel parameters.
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

        # compute dK/dtheta_j, dK/dtheta_i, d2K/(dtheta_j dtheta_i)
        dj = np.empty((x.size, x.size))
        di = np.empty((x.size, x.size))
        dji = np.empty((x.size, x.size))
        for i in xrange(x.size):
            for j in xrange(x.size):
                diff = x[i] - x[j]
                dj[i, j] = dK_dthetaj(theta, diff)
                di[i, j] = dK_dthetai(theta, diff)
                dji[i, j] = d2K_dthetaji(theta, diff)

        dKidi = np.dot(-Ki, np.dot(di, Ki))
        dKidj = np.dot(-Ki, np.dot(dj, Ki))
        d2Kidji = np.dot(-Ki, np.dot(dji, Ki))

        k = np.dot(dKidi, dKidj) + d2Kidji + np.dot(dKidj, dKidi)
        t0 = 0.5 * np.dot(y.T, np.dot(k, y))
        t1 = -0.5 * np.trace(np.dot(dKidi, dj) + np.dot(Ki, dji))
        dd = t0 + t1

        return dd

    def lh(self, theta, x, y):
        """Computes the marginal log likelihood of the kernel.

        This is computing Eq. 5.8 of Rasmussen & Williams (2006):

        $$\log{p(\mathbf{y} | X,\mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \frac{1}{2}\log{|K_y|} - \frac{n}{2}\log{2\pi}

        Parameters
        ----------
        theta : tuple
            The kernel parameters.
        x : numpy.ndarray with shape (n,)
            The given input values
        y : numpy.ndarray with shape (n,)
            The given output values

        Returns
        -------
        out : float
            The marginal log likelihood of the data given the parameters

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        # invert K and compute determinant
        K = self.Kxx(theta, x)
        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        Li = np.linalg.inv(L)
        Ki = np.dot(Li.T, Li)
        logdetK = 2 * np.log(np.diag(L)).sum()

        # compute the marginal log likelihood
        t1 = -0.5 * np.dot(np.dot(y.T, Ki), y)
        t2 = -0.5 * logdetK
        t3 = -0.5 * x.size * np.log(2 * np.pi)
        mll = np.clip(t1 + t2 + t3, MIN_LOG, MAX_LOG)

        return mll

    def neg_lh(self, *args, **kwargs):
        llh = self.lh(*args, **kwargs)
        return -llh

    def jacobian(self, theta, x, y):
        """Computes the Jacobian of the marginal log likelihood of
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
            The Jacobian of the marginal negative log likelihood of the
            data given the parameters

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        # invert K
        K = self.Kxx(theta, x)
        try:
            Li = np.linalg.inv(cholesky(K))
        except np.linalg.LinAlgError:
            return np.zeros(len(self.dK_dtheta)) - np.inf
        Ki = np.dot(Li.T, Li)

        dmll = np.empty(len(self.dK_dtheta))
        for i in xrange(dmll.size):
            dK_dthetaj = self.dK_dtheta[i]
            dmll[i] = self._d_dthetaj(Ki, dK_dthetaj, theta, x, y)

        dmll = np.clip(dmll, MIN_LOG, MAX_LOG)
        return dmll

    def neg_jacobian(self, *args, **kwargs):
        lj = self.jacobian(*args, **kwargs)
        return -lj

    def hessian(self, theta, x, y):
        """Computes the Hessian of the marginal log likelihood of the
        kernel.

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
            The Hessian of the marginal negative log likelihood of the
            data given the parameters

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        # invert K
        K = self.Kxx(theta, x)
        try:
            Li = np.linalg.inv(cholesky(K))
        except np.linalg.LinAlgError:
            return -np.inf
        Ki = np.dot(Li.T, Li)

        n = len(self.dK_dtheta)
        ddmll = np.empty((n, n))
        for i in xrange(n):
            dK_dthetai = self.dK_dtheta[i]
            for j in xrange(n):
                dK_dthetaj = self.dK_dtheta[j]
                d2K = self.d2K_dtheta2[j][i]
                ddmll[j, i] = self._d2_dthetajdthetai(
                    Ki, dK_dthetaj, dK_dthetai, d2K, theta, x, y)

        ddmll = np.clip(ddmll, MIN_LOG, MAX_LOG)
        return ddmll

    def maximize(self, x, y,
                 hmin=1e-8, hmax=None,
                 wmin=1e-8, wmax=None,
                 smin=0, smax=None,
                 pmin=0, pmax=None,
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

        bounds = []
        if self.h is None:
            bounds.append((hmin, hmax))
        if self.w is None:
            bounds.append((wmin, wmax))
        if self.s is None:
            bounds.append((smin, smax))
        if self.p is None:
            bounds.append((pmin, pmax))

        args = np.empty((ntry, len(bounds)))
        fval = np.empty(ntry)

        for i in xrange(ntry):
            # randomize starting parameter values
            p0 = []
            if self.h is None:
                p0.append(np.random.uniform(hmin, np.max(np.abs(y))*2))
            if self.w is None:
                p0.append(np.random.uniform(wmin, 2*np.pi))
            if self.s is None:
                p0.append(np.random.uniform(smin, np.sqrt(np.var(y))))
            if self.p is None:
                p0.append(np.random.uniform(pmin, 2*np.pi))

            method = "L-BFGS-B"
            if verbose:
                print "      p0 = %s" % (p0,)

            # run mimization function
            # try:
            popt = opt.minimize(
                fun=self.neg_lh,
                x0=p0,
                args=(x, y),
                jac=self.neg_jacobian,
                method=method,
                bounds=bounds
            )

            # get results of the optimization
            args[i] = popt['x']
            fval[i] = popt['fun']

            if verbose:
                print "      -MLL(%s) = %f" % (args[i], fval[i])

        # choose the parameters that give the best MLL
        if args is None or fval is None:
            raise RuntimeError("Could not find MLII parameter estimates")
        best = np.argmin(fval)
        params = self.kernel_params(args[best])

        return params

    def dm_dw(self, theta, x, y, xo):
        """Compute the partial derivative of a GP mean with respect to
        w, the input scale parameter.

        The analytic form is:
        $\frac{\partial K(x_*, x)}{\partial w}K_y^{-1}\mathbf{y} - K(x_*, x)K_y^{-1}\frac{\partial K(x, x)}{\partial w}K_y^{-1}\mathbf{y}$

        Where $K_y=K(x, x) + s^2I$

        """

        if self.w is not None:
            raise ValueError("w is not a free parameter")

        # compute Kxx and Kxox
        Kxx = self.Kxx(theta, x)
        Kxox = self.make_kernel(theta=theta, jit=False)(xo, x)

        # invert K
        try:
            L = cholesky(Kxx)
        except np.linalg.LinAlgError:
            return np.nan
        Li = np.linalg.inv(L)
        inv_Kxx = np.dot(Li.T, Li)

        # get the partial derivative function
        if self.h is None:
            dK_dw = self.dK_dtheta[1]
        else:
            dK_dw = self.dK_dtheta[0]

        # compute dK/dw matrix for x, x
        dKxx = np.empty((x.size, x.size))
        for i in xrange(x.size):
            for j in xrange(x.size):
                diff = x[i] - x[j]
                dKxx[i, j] = dK_dw(theta, diff)

        # compute dK/dw matrix for xo, x
        dKxox = np.empty((xo.size, x.size))
        for i in xrange(xo.size):
            for j in xrange(x.size):
                diff = xo[i] - x[j]
                dKxox[i, j] = dK_dw(theta, diff)

        # compute the two terms of the derivative and combine them
        v = np.dot(inv_Kxx, y)
        t0 = np.dot(dKxox, v)
        t1 = np.dot(Kxox, np.dot(-inv_Kxx, np.dot(dKxx, v)))
        dm = t0 + t1
        return dm
