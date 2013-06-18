import numpy as np

from numpy import dot, log, pi, exp, trace
from numpy.linalg import inv


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


class GP(object):

    def __init__(self, K, x, y, xo, s=0):
        self.K = K
        self.x = x
        self.y = y
        self.xo = xo
        self.s = s

    @property
    def params(self):
        return tuple(list(self.K.params) + [self._s])

    @params.setter
    def params(self, val):
        params = self.params
        if params[:-1] != val[:-1]:
            self.K.params = val[:-1]
            self._Kxx = None
            self._inv_Kxx = None
            self._inv_Kxx_y = None
            self._Kxoxo = None
            self._Kxxo = None
            self._Kxox = None
            self._m = None
            self._C = None
        if params[-1] != val[-1]:
            self.s = val[-1]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._Kxx = None
        self._inv_Kxx = None
        self._inv_Kxx_y = None
        self._Kxxo = None
        self._Kxox = None
        self._m = None
        self._C = None
        self._x = val.copy()

    @property
    def xo(self):
        return self._xo

    @xo.setter
    def xo(self, val):
        self._Kxoxo = None
        self._Kxxo = None
        self._Kxox = None
        self._m = None
        self._C = None
        self._xo = val.copy()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._inv_Kxx_y = None
        self._m = None
        self._C = None
        self._y = val.copy()

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, val):
        self._Kxx = None
        self._m = None
        self._C = None
        self._s = val

    @property
    def Kxx(self):
        if self._Kxx is None:
            self._Kxx = self.K(self._x, self._x)
            self._Kxx += np.eye(self._x.size) * (self._s ** 2)
            if np.isnan(self._Kxx).any():
                print self.K.params
                raise ArithmeticError("Kxx contains invalid values")
        return self._Kxx

    @property
    def inv_Kxx(self):
        if self._inv_Kxx is None:
            Li = inv(cholesky(self.Kxx))
            self._inv_Kxx = dot(Li.T, Li)
        return self._inv_Kxx

    @property
    def Kxoxo(self):
        if self._Kxoxo is None:
            self._Kxoxo = self.K(self._xo, self._xo)
        return self._Kxoxo

    @property
    def Kxxo(self):
        if self._Kxxo is None:
            self._Kxxo = self.K(self._x, self._xo)
        return self._Kxxo

    @property
    def Kxox(self):
        if self._Kxox is None:
            self._Kxox = self.K(self._xo, self._x)
        return self._Kxox

    @property
    def inv_Kxx_y(self):
        if self._inv_Kxx_y is None:
            self._inv_Kxx_y = dot(self.inv_Kxx, self.y)
        return self._inv_Kxx_y

    @property
    def m(self):
        """Predictive mean of the GP.

        This is computing Eq. 2.23 of Rasmussen & Williams (2006):

        $$\mathbf{m}=K(X_*, X)[K(X, X) + \sigma_n^2]^{-1}\mathbf{y}$$

        """
        if self._m is None:
            self._m = dot(self.Kxox, self.inv_Kxx_y)
        return self._m

    @property
    def C(self):
        """Predictive covariance of the GP.

        This is computing Eq. 2.24 of Rasmussen & Williams (2006):

        $$\mathbf{C}=K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2]^{-1}K(X, X_*)$$

        """
        if self._C is None:
            self._C = self.Kxoxo - dot(self.Kxox, dot(self.inv_Kxx, self.Kxxo))
        return self._C

    def log_lh(self):
        """The log likelihood of y given x and theta.

        This is computing Eq. 5.8 of Rasmussen & Williams (2006):

        $$\log{p(\mathbf{y} | X \mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \frac{1}{2}\log{|K_y|}-\frac{n}{2}\log{2\pi}$$

        """

        y, K = self._y, self.Kxx
        sign, logdet = np.linalg.slogdet(K)
        if sign != 1:
            return -np.inf

        try:
            Kiy = self.inv_Kxx_y
        except np.linalg.LinAlgError:
            return -np.inf

        data_fit = -0.5 * dot(y.T, Kiy)
        complexity_penalty = -0.5 * logdet
        constant = -0.5 * y.size * log(2 * pi)
        llh = data_fit + complexity_penalty + constant
        return llh

    def lh(self):
        return np.exp(self.log_lh())

    def dloglh_dtheta(self):
        """Compute the partial derivative of the marginal log likelihood with
        respect to its parameters `\theta`.

        See Eq. 5.9 of Rasmussen & Williams (2006).

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        x, y = self._x, self._y
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            return np.array([-np.inf for p in self.params])

        nparam = len(self.params)

        # compute kernel jacobian
        dK_dtheta = np.empty((nparam, y.size, y.size))
        dK_dtheta[:-1] = self.K.jacobian(x, x)
        dK_dtheta[-1] = np.eye(y.size) * 2 * self._s

        dloglh = np.empty(nparam)
        for i in xrange(dloglh.size):
            k = np.dot(Ki, dK_dtheta[i])
            t0 = 0.5 * dot(y.T, np.dot(k, self.inv_Kxx_y))
            t1 = -0.5 * np.trace(k)
            dloglh[i] = t0 + t1

        return dloglh

    def dlh_dtheta(self):
        """Compute the partial derivative of the marginal likelihood with
        respect to its parameters `\theta`.

        See Eq. 5.9 of Rasmussen & Williams (2006).

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        x, y, K, Ki = self._x, self._y, self.Kxx, self.inv_Kxx
        Kiy = self.inv_Kxx_y
        nparam = len(self.params)

        dK_dtheta = np.empty((nparam, y.size, y.size))
        dK_dtheta[:-1] = self.K.jacobian(x, x)
        dK_dtheta[-1] = np.eye(y.size) * 2 * self._s

        lh = self.lh()
        dlh = np.empty(nparam)
        for i in xrange(dlh.size):
            KidK = dot(Ki, dK_dtheta[i])
            t0 = dot(y.T, dot(KidK, Kiy))
            t1 = trace(KidK)
            dlh[i] = 0.5 * lh * (t0 - t1)

        return dlh

    def d2lh_dtheta2(self):
        y, x, K, Ki = self._y, self._x, self.Kxx, self.inv_Kxx
        Kiy = self.inv_Kxx_y
        nparam = len(self.params)

        # first kernel derivatives
        dK = np.empty((nparam, y.size, y.size))
        dK[:-1] = self.K.jacobian(x, x)
        dK[-1] = np.eye(y.size) * 2 * self._s
        dKi = [dot(-Ki, dot(dK[i], Ki)) for i in xrange(nparam)]

        # second kernel derivatives
        d2K = np.zeros((nparam, nparam, y.size, y.size))
        d2K[:-1, :-1] = self.K.hessian(x, x)
        d2K[-1, -1] = np.eye(y.size) * 2

        # likelihood
        lh = self.lh()
        # first derivative of the likelihood
        dlh = self.dlh_dtheta()

        d2lh = np.empty((nparam, nparam))
        for i in xrange(nparam):
            KidK_i = dot(Ki, dK[i])
            ydKi_iy = dot(y.T, dot(KidK_i, Kiy))
            ydKi_iy_tr = ydKi_iy - trace(KidK_i)

            for j in xrange(nparam):
                dKi_jdK_i = dot(dKi[j], dK[i])
                d_ydKi_iy = (
                    dot(y.T, dot(dKi_jdK_i, Kiy)) +
                    dot(Kiy.T, dot(d2K[i, j], Kiy)) +
                    dot(Kiy.T, dot(dK[i], dot(dKi[j], y))))
                d_tr = trace(dKi_jdK_i + dot(Ki, d2K[i, j]))

                t0 = dlh[j] * ydKi_iy_tr
                t1 = lh * (d_ydKi_iy - d_tr)
                d2lh[i, j] = 0.5 * (t0 + t1)

        return d2lh

    def copy(self):
        new_gp = GP(self.K.copy(), self.x, self.y, self.xo)
        copy = lambda x: None if x is None else x.copy()
        new_gp._Kxx = copy(self._Kxx)
        new_gp._inv_Kxx = copy(self._inv_Kxx)
        new_gp._inv_Kxx_y = copy(self._inv_Kxx_y)
        new_gp._Kxoxo = copy(self._Kxoxo)
        new_gp._Kxxo = copy(self._Kxxo)
        new_gp._Kxox = copy(self._Kxox)
        new_gp._m = copy(self._m)
        new_gp._C = copy(self._C)
        return new_gp
