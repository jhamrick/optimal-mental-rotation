import numba

import numpy as np
from numpy import exp

import sympy as sym
from sympy.functions.special.delta_functions import DiracDelta as delta

from base_kernel import BaseKernel


class GaussianKernel(object):
    """Represents a gaussian kernel function, of the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{(x_1-x_2)^2}{2w^2})$$

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    __metaclass__ = BaseKernel

    def __init__(self, h, w, s):
        """Create a GaussianKernel object at specific parameter values.

        Parameters
        ----------
        h : number
            Output scale kernel parameter
        w : number
            Input scale (Gaussian standard deviation) kernel parameter
        s : number
            Observation noise standard deviation

        """

        self.h = h
        self.w = w
        self.s = s

    @property
    def sym_K(self):
        h = self._sym_h
        w = self._sym_w
        s = self._sym_s
        d = self._sym_d

        h2 = h ** 2
        w2 = w ** 2
        d2 = d ** 2
        obs = (s ** 2) * delta(d)

        f = h2 * sym.exp(-d2 / (2.0 * w2)) + obs
        return f

    def copy(self):
        return GaussianKernel(*self.params)

    @property
    def params(self):
        return (self.h, self.w, self.s)

    @params.setter
    def params(self, val):
        self.h, self.w, self.s = val

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _K(x1, x2, h, w, s):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1.0
                else:
                    dd = 0.0
                Kxx[i, j] = h**2*exp(-d**2/(2.0*w**2)) + s**2*dd
        return Kxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_dh(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*h*exp(-d**2/(2.0*w**2))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_dw(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = d**2*h**2*exp(-d**2/(2.0*w**2))/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_ds(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1.0
                else:
                    dd = 0.0
                dKxx[i, j] = 2.0*s*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _jacobian(x1, x2, h, w, s):
        dKxx = np.empty((3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1.0
                else:
                    dd = 0.0
                dKxx[0, i, j] = 2.0*h*exp(-d**2/(2.0*w**2))
                dKxx[1, i, j] = d**2*h**2*exp(-d**2/(2.0*w**2))/w**3
                dKxx[2, i, j] = 2.0*s*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhdh(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*exp(-d**2/(2*w**2))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhdw(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*d**2*h*exp(-d**2/(2.0*w**2))/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhds(x1, x2, h, w, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwdh(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*d**2*h*exp(-d**2/(2.0*w**2))/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwdw(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = d**4*h**2*exp(-d**2/(2.0*w**2))/w**6 - 3.0*d**2*h**2*exp(-d**2/(2.0*w**2))/w**4
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwds(x1, x2, h, w, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dsdh(x1, x2, h, w, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dsdw(x1, x2, h, w, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dsds(x1, x2, h, w, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1.0
                else:
                    dd = 0.0
                dKxx[i, j] = 2.0*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _hessian(x1, x2, h, w, s):
        dKxx = np.empty((3, 3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1.0
                else:
                    dd = 0.0

                # h
                dKxx[0, 0, i, j] = 2.0*exp(-d**2/(2.0*w**2))
                dKxx[0, 1, i, j] = 2.0*d**2*h*exp(-d**2/(2.0*w**2))/w**3
                dKxx[0, 2, i, j] = 0

                # w
                dKxx[1, 0, i, j] = 2.0*d**2*h*exp(-d**2/(2.0*w**2))/w**3
                dKxx[1, 1, i, j] = d**4*h**2*exp(-d**2/(2.0*w**2))/w**6 - 3.0*d**2*h**2*exp(-d**2/(2.0*w**2))/w**4
                dKxx[1, 2, i, j] = 0

                # s
                dKxx[2, 0, i, j] = 0
                dKxx[2, 1, i, j] = 0
                dKxx[2, 2, i, j] = 2.0*dd

        return dKxx
