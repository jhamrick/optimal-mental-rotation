from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport exp, log, fmax, copysign, fabs, M_PI

cdef inv = np.linalg.inv
cdef slogdet = np.linalg.slogdet

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps


def mvn_logpdf(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=2] C, Z=None):
    cdef np.ndarray[DTYPE_t, ndim=2] Ci = inv(C)
    cdef np.ndarray[DTYPE_t, ndim=2] diff = x - m
    cdef np.ndarray[DTYPE_t, ndim=1] pdf
    cdef int n, d, i, j, k
    cdef DTYPE_t t0, t1, sign, v

    if Z is None:
        Z = -0.5 * slogdet(C)[1]

    n = x.shape[0]
    d = x.shape[1]
    t0 = log(2*M_PI)*(-d/2.) + Z
    pdf = np.empty(n, dtype=DTYPE)

    for i in xrange(n):
        t1 = 0
        for j in xrange(d):
            for k in xrange(d):
                sign = copysign(1, diff[i, j]) * copysign(1, Ci[j, k]) * copysign(1, diff[i, k])
                v = log(fabs(diff[i, j])) + log(fabs(Ci[j, k])) + log(fabs(diff[i, k]))
                if v > MIN:
                    t1 += copysign(exp(v), sign)
        pdf[i] = t0 - 0.5*t1

    return pdf


def improve_covariance_conditioning(np.ndarray[DTYPE_t, ndim=2] M):
    cdef DTYPE_t sqd_jitters = fmax(EPS, np.max(M)) * 1e-4
    cdef int n, i
    n = M.shape[0]
    for i in xrange(n):
        M[i, i] += sqd_jitters


def gaussint1(np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h, DTYPE_t w, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    cdef np.ndarray[DTYPE_t, ndim=2] W
    cdef np.ndarray[DTYPE_t, ndim=2] vec
    cdef int n, d, i, j

    n = x.shape[0]
    d = x.shape[1]
    W = np.empty((d, d))

    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W[i, j] = cov[i, j] + w**2
            else:
                W[i, j] = cov[i, j]

    vec = np.empty((1, n), dtype=DTYPE)
    vec[0] = h**2 * np.exp(mvn_logpdf(x, mu, W))
    return vec


def gaussint2(np.ndarray[DTYPE_t, ndim=3] x1, np.ndarray[DTYPE_t, ndim=3] x2, DTYPE_t h1, DTYPE_t w1, DTYPE_t h2, DTYPE_t w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    cdef np.ndarray[DTYPE_t, ndim=2] Wa
    cdef np.ndarray[DTYPE_t, ndim=2] Wb
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef int n1, n2, d1, d2, i, j
    cdef DTYPE_t ha, hb

    n1 = x1.shape[0]
    d1 = x1.shape[1]
    n2 = x2.shape[0]
    d2 = x1.shape[1]

    ha = h1 ** 2
    hb = h2 ** 2

    Wa = np.eye(d1, dtype=DTYPE) * w1
    Wb = np.eye(d2, dtype=DTYPE) * w2

    i1, i2 = np.meshgrid(np.arange(n1), np.arange(n2))
    x = np.concatenate([x1[i1.T], x2[i2.T]], axis=2)
    m = np.concatenate([mu, mu])
    C = np.concatenate([
        np.concatenate([Wa + cov, cov], axis=1),
        np.concatenate([cov, Wb + cov], axis=1)
    ], axis=0)

    mat = ha * hb * np.exp(mvn_logpdf(x.reshape((-1, 2)), m, C)).reshape((n1, n2))
    return mat


def gaussint3(x, h1, w1, h2, w2, mu, cov):
    n, d = x.shape

    ha = h1 ** 2
    hb = h2 ** 2
    Wa = (np.array(w1) * np.eye(d)) ** 2
    Wb = (np.array(w2) * np.eye(d)) ** 2

    G = np.dot(cov, inv(Wa + cov))
    Gi = inv(G)
    C = Wb + 2*cov - 2*np.dot(G, cov)

    N1 = np.exp(mvn_logpdf(x, mu, cov + Wa))

    x2 = x[:, None] - x[None, :]
    mu2 = np.zeros(d)
    cov2 = np.dot(np.dot(Gi, C), Gi)
    Z = slogdet(C)[1] * -0.5
    N2 = np.exp(mvn_logpdf(x2.reshape((-1, mu2.size)), mu2, cov2, Z=Z)).reshape(x2.shape[:-1])

    mat = ha**2 * hb * N2 * N1[:, None] * N1[None, :]
    return mat

def gaussint4(x, h1, w1, h2, w2, mu, cov):
    n, d = x.shape

    ha = h1 ** 2
    hb = h2 ** 2
    Wa = (np.array(w1) * np.eye(d)) ** 2
    Wb = (np.array(w2) * np.eye(d)) ** 2
    
    C0 = Wa + 2*cov
    C1 = Wb + cov - np.dot(np.dot(cov, inv(C0)), cov)

    N0 = np.exp(mvn_logpdf(np.zeros((1, d)), np.zeros(d), C0))
    N1 = np.exp(mvn_logpdf(x, mu, C1))
    
    vec = (ha * hb * N0 * N1)[None]
    return vec

def gaussint5(d, h, w, mu, cov):
    h2 = h ** 2
    W = (np.array(w) * np.eye(d)) ** 2
    const = float(h2 * np.exp(mvn_logpdf(
        np.zeros((1, d)),
        np.zeros(d),
        W + 2*cov)))
    return const

def dtheta_consts(x, w1, w2, mu, L):
    n, d = x.shape
    I = np.eye(d)
    Wl = (w1 ** 2) * I
    Wtl = (w2 ** 2) * I
    iWtl = inv(Wtl)

    A = np.dot(L, inv(Wtl + L))
    B = L - np.dot(A, L)
    C = np.dot(B, inv(B + Wl))
    CA = np.dot(C, A)
    BCB = B - np.dot(C, B)

    xsubmu = x - mu
    mat_const = np.empty((n, n, d))
    vec_const = np.empty((n, d))
    c = -0.5*iWtl

    m1 = np.dot(A - I, xsubmu.T).T
    m2a = np.dot(A - CA - I, xsubmu.T).T
    m2b = np.dot(C, xsubmu.T).T
    
    for i in xrange(n):
        vec_const[i] = np.diag(
            c * (I + np.dot(B - np.dot(m1[i], m1[i].T), iWtl)))

        for j in xrange(n):
            m2 = m2a[i] + m2b[j]
            mat_const[i, j] = np.diag(
                c * (I + np.dot(BCB - np.dot(m2, m2.T), iWtl)))
            
    return mat_const, vec_const
