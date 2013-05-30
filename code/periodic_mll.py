import numba
import numpy as np
from numpy import exp, sin, cos


def K(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            if d == 0:
                dd = 1
            else:
                dd = 0
            dj[i, j] = h**2*exp(-2.0*sin(0.5*d/p)**2/w**2) + s**2*dd
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def dK_dh(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 2*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def dK_dw(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def dK_ds(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            if d == 0:
                dd = 1
            else:
                dd = 0
            dj[i, j] = 2*s*dd
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def dK_dp(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dhdh(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 2*exp(-2.0*sin(0.5*d/p)**2/w**2)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dhdw(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dhds(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dhdp(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dwdh(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dwdw(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dwds(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dwdp(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dsdh(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dsdw(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dsds(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            if d == 0:
                dd = 1
            else:
                dd = 0
            dj[i, j] = 2*dd
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dsdp(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dpdh(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dpdw(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dpds(x1, x2, h, w, p, s):
    dj = np.zeros((x1.size, x2.size))
    return dj


@numba.jit('f8[:,:](f8[:], f8[:], f8, f8, f8, f8)', warn=False)
def d2K_dpdp(x1, x2, h, w, p, s):
    dj = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            d = x1[i] - x2[j]
            dj[i, j] = d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)
    return dj


jacobian = (
    dK_dh,
    dK_dw,
    dK_ds,
    dK_dp
)

hessian = (
    (d2K_dhdh,
     d2K_dhdw,
     d2K_dhds,
     d2K_dhdp),

    (d2K_dwdh,
     d2K_dwdw,
     d2K_dwds,
     d2K_dwdp),

    (d2K_dsdh,
     d2K_dsdw,
     d2K_dsds,
     d2K_dsdp),

    (d2K_dpdh,
     d2K_dpdw,
     d2K_dpds,
     d2K_dpdp)
)
