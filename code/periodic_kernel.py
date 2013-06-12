import numba

import numpy as np
from numpy import exp, sin, cos

import sympy as sym
from sympy.functions.special.delta_functions import DiracDelta as delta

from base_kernel import BaseKernel


class PeriodicKernel(object):
    """Represents a periodic kernel function, of the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{2\sin^2(\frac{x_1-x_2}{2p})}{w^2})$$

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    __metaclass__ = BaseKernel

    _sym_h = sym.Symbol('h')
    _sym_w = sym.Symbol('w')
    _sym_p = sym.Symbol('p')
    _sym_s = sym.Symbol('s')
    _sym_d = sym.Symbol('d')

    def __init__(self, h, w, p, s):
        """Create a PeriodicKernel object at specific parameter values.

        Parameters
        ----------
        h : number
            Output scale kernel parameter
        w : number
            Input scale (Gaussian standard deviation) kernel parameter
        p : number
            Period kernel parameter
        s : number
            Observation noise standard deviation

        """

        self.h = h
        self.w = w
        self.p = p
        self.s = s

    @property
    def sym_K(self):
        h = self._sym_h
        w = self._sym_w
        p = self._sym_p
        s = self._sym_s
        d = self._sym_d

        h2 = h ** 2
        w2 = w ** 2
        obs = (s ** 2) * delta(d)

        f = h2 * sym.exp(-2.*(sym.sin(d / (2.*p)) ** 2) / w2) + obs
        return f

    def copy(self):
        return PeriodicKernel(*self.params)

    @property
    def params(self):
        return (self.h, self.w, self.p, self.s)

    @params.setter
    def params(self, val):
        self.h, self.w, self.p, self.s = val

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _K(x1, x2, h, w, p, s):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0
                Kxx[i, j] = h**2*exp(-2.0*sin(0.5*d/p)**2/w**2) + s**2*dd
        return Kxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _dK_dh(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _dK_dw(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _dK_dp(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _dK_ds(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0
                dKxx[i, j] = 2.0*s*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _jacobian(x1, x2, h, w, p, s):
        dKxx = np.empty((4, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0
                dKxx[0, i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
                dKxx[1, i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[2, i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
                dKxx[3, i, j] = 2.0*s*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dhdh(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dhdw(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dhdp(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dhds(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dwdh(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dwdw(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dwdp(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dwds(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dpdh(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dpdw(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dpdp(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dpds(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dsdh(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dsdw(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dsdp(x1, x2, h, w, p, s):
        dKxx = np.zeros((x1.size, x2.size))
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _d2K_dsds(x1, x2, h, w, p, s):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0
                dKxx[i, j] = 2.0*dd
        return dKxx

    @staticmethod
    @numba.jit('f8[:,:,:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
    def _hessian(x1, x2, h, w, p, s):
        dKxx = np.empty((4, 4, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0

                # h
                dKxx[0, 0, i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
                dKxx[0, 1, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[0, 2, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
                dKxx[0, 3, i, j] = 0

                # w
                dKxx[1, 0, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[1, 1, i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
                dKxx[1, 2, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
                dKxx[1, 3, i, j] = 0

                # p
                dKxx[2, 0, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
                dKxx[2, 1, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
                dKxx[2, 2, i, j]= d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)
                dKxx[2, 3, i, j] = 0

                # s
                dKxx[3, 0, i, j] = 0
                dKxx[3, 1, i, j] = 0
                dKxx[3, 2, i, j] = 0
                dKxx[3, 3, i, j] = 2.0*dd

        return dKxx
