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

    def __init__(self, K, x, y, xo):
        self.K = K
        self.x = x
        self.y = y
        self.xo = xo

    @property
    def params(self):
        return self.K.params

    @params.setter
    def params(self, val):
        if self.K.params == val:
            return
        self.K.params = val
        self._Kxx = None
        self._inv_Kxx = None
        self._Kxoxo = None
        self._Kxxo = None
        self._Kxox = None
        self._m = None
        self._C = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._Kxx = None
        self._inv_Kxx = None
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
        self._m = None
        self._C = None
        self._y = val.copy()

    @property
    def Kxx(self):
        if self._Kxx is None:
            self._Kxx = self.K(self._x, self._x)
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
    def m(self):
        """Predictive mean of the GP.

        This is computing Eq. 2.23 of Rasmussen & Williams (2006):

        $$\mathbf{m}=K(X_*, X)[K(X, X) + \sigma_n^2]^{-1}\mathbf{y}$$

        """
        if self._m is None:
            Ki, y = self.inv_Kxx, self._y
            self._m = dot(self.Kxox, dot(Ki, y))
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

        Ki = self.inv_Kxx
        data_fit = -0.5 * dot(y.T, dot(Ki, y))
        complexity_penalty = -0.5 * logdet
        constant = -0.5 * y.size * log(2 * pi)
        llh = data_fit + complexity_penalty + constant
        return llh

    def lh(self):
        return np.exp(self.log_lh())

    def dloglh_dtheta(self):
        y = self._y
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            return np.array([-np.inf for p in self.params])

        # compute kernel jacobian
        dK_dtheta = self.K.jacobian(self.x, self.x)

        dloglh = np.empty(dK_dtheta.shape[0])
        for i in xrange(dloglh.size):
            k = np.dot(Ki, dK_dtheta[i])
            t0 = 0.5 * dot(y.T, np.dot(k, np.dot(Ki, y)))
            t1 = -0.5 * np.trace(k)
            dloglh[i] = t0 + t1

        return dloglh

    def dlh_dtheta(self):
        y, K, Ki = self._y, self.Kxx, self.inv_Kxx
        n = y.size

        dK_dtheta = self.K.jacobian(self.x, self.x)
        yKiy = dot(y.T, dot(Ki, y))
        invsqrtdet = np.linalg.det(K) ** (-1 / 2.)
        t0 = -0.5 * exp(-0.5 * yKiy) * (2 * pi) ** (-n / 2.) * invsqrtdet

        dlh = np.empty(dK_dtheta.shape[0])
        for i in xrange(dlh.size):
            dinvK_dtheta = dot(-Ki, dK_dtheta[i], Ki)
            t1 = dot(y.T, dot(dinvK_dtheta, y))
            t2 = trace(dot(Ki, dK_dtheta[i]))
            dlh[i] = t0 * (t1 + t2)

        return dlh

    def d2lh_dtheta2(self):
        y, K, Ki = self._y, self.Kxx, self.inv_Kxx
        n = y.size

        dK_dtheta = self.K.jacobian(self.x, self.x)
        dKinv_dtheta = [dot(-Ki, dot(dK_dtheta[i], Ki))
                        for i in xrange(dK_dtheta.shape[0])]
        tr = [trace(dot(Ki, dK_dtheta[i]))
              for i in xrange(dK_dtheta.shape[0])]
        d2K_dtheta2 = self.K.hessian(self.x, self.x)
        invsqrtdet = np.linalg.det(K) ** (-1 / 2.)

        yKinv = dot(y.T, Ki)
        Kinvy = dot(Ki, y)

        A = exp(-0.5 * dot(y.T, dot(Ki, y)))
        B = invsqrtdet
        const = -0.5 * (2 * pi) ** (-n / 2.)

        d2lh = np.empty((dK_dtheta.shape[0], dK_dtheta.shape[0]))
        for i in xrange(d2lh.shape[0]):
            C = dot(y.T, dot(dKinv_dtheta[i], y))
            D = tr[i]

            for j in xrange(d2lh.shape[1]):
                dA = -0.5 * dot(y.T, dot(dKinv_dtheta[j], y)) * A
                dB = -0.5 * invsqrtdet * tr[j]
                dC = (-dot(y.T, dot(dKinv_dtheta[j], dot(dK_dtheta[i], Kinvy)))
                      - dot(yKinv, dot(d2K_dtheta2[i][j], Kinvy))
                      - dot(yKinv, dot(dK_dtheta[i], dot(dKinv_dtheta[j], y))))
                dD = trace(dot(dKinv_dtheta[j], dK_dtheta[i]) +
                           dot(Ki, d2K_dtheta2[i][j]))

                d2lh[i, j] = const * (
                    dA * B * (C + D) +
                    A * dB * (C + D) +
                    A * B * (dC + dD))

        return d2lh

    def copy(self):
        new_gp = GP(self.K.copy(), self.x, self.y, self.xo)
        copy = lambda x: None if x is None else x.copy()
        new_gp._Kxx = copy(self._Kxx)
        new_gp._inv_Kxx = copy(self._inv_Kxx)
        new_gp._Kxoxo = copy(self._Kxoxo)
        new_gp._Kxxo = copy(self._Kxxo)
        new_gp._Kxox = copy(self._Kxox)
        new_gp._m = copy(self._m)
        new_gp._C = copy(self._C)
