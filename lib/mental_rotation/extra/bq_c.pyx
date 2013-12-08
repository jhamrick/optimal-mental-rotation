from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport exp, log, fmax, copysign, fabs, M_PI

cdef inv = np.linalg.inv
cdef slogdet = np.linalg.slogdet
cdef dot = np.dot

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

    G = dot(cov, inv(Wa + cov))
    Gi = inv(G)
    C = Wb + 2*cov - 2*dot(G, cov)

    N1 = np.exp(mvn_logpdf(x, mu, cov + Wa))

    x2 = x[:, None] - x[None, :]
    mu2 = np.zeros(d)
    cov2 = dot(dot(Gi, C), Gi)
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
    C1 = Wb + cov - dot(dot(cov, inv(C0)), cov)

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

    A = dot(L, inv(Wtl + L))
    B = L - dot(A, L)
    C = dot(B, inv(B + Wl))
    CA = dot(C, A)
    BCB = B - dot(C, B)

    xsubmu = x - mu
    mat_const = np.empty((n, n, d))
    vec_const = np.empty((n, d))
    c = -0.5*iWtl

    m1 = dot(A - I, xsubmu.T).T
    m2a = dot(A - CA - I, xsubmu.T).T
    m2b = dot(C, xsubmu.T).T
    
    for i in xrange(n):
        vec_const[i] = np.diag(
            c * (I + dot(B - dot(m1[i], m1[i].T), iWtl)))

        for j in xrange(n):
            m2 = m2a[i] + m2b[j]
            mat_const[i, j] = np.diag(
                c * (I + dot(BCB - dot(m2, m2.T), iWtl)))
            
    return mat_const, vec_const

def Z_mean(x_s, x_sc, alpha_l, alpha_del, h_s, w_s, h_dc, w_dc, mu, cov, gamma):
    ns, d = x_s.shape
    nc, d = x_sc.shape

    ## First term
    # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
    int_K_l = gaussint1(x_s, h_s, w_s, mu, cov)
    E_m_l = float(dot(int_K_l, alpha_l))
    assert E_m_l > 0

    ## Second term
    # E[m_l*m_del | x_s, x_c] = alpha_del(x_sc)' *
    #     int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
    #     alpha_l(x_s)
    int_K_del_K_l = gaussint2(
        x_sc, x_s, h_dc, w_dc, h_s, w_s, mu, cov)
    E_m_l_m_del = float(dot(dot(
        alpha_del.T, int_K_del_K_l), alpha_l))
    
    ## Third term
    # E[m_del | x_sc] = (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
    int_K_del = gaussint1(x_sc, h_dc, w_dc, mu, cov)
    E_m_del = float(dot(int_K_del, alpha_del))
    
    # put the three terms together
    m_Z = E_m_l + E_m_l_m_del + gamma * E_m_del

    return m_Z


def Z_var(x_s, alpha_l, alpha_tl, inv_L_tl, inv_K_tl, dK_tl_dw, Cw, h_l, w_l, h_tl, w_tl, mu, cov, gamma):
    ns, d = x_s.shape

    ## First term
    # E[m_l C_tl m_l | x_s] = alpha_l(x_s)' *
    #    int int K_l(x_s, x) K_tl(x, x') K_l(x', x_s) p(x) p(x') dx dx' *
    #    alpha_l(x_s) - beta(x_s)'beta(x_s)
    # Where beta is defined as:
    # beta(x_s) = inv(L_tl(x_s, x_s)) *
    #    int K_tl(x_s, x) K_l(x, x_s) p(x) dx *
    #    alpha_l(x_s)
    int_K_l_K_tl_K_l = gaussint3(
        x_s, h_l, w_l, h_tl, w_tl, mu, cov)
    int_K_tl_K_l_mat = gaussint2(
        x_s, x_s, h_tl, w_tl, h_l, w_l, mu, cov)
    beta = dot(dot(inv_L_tl, int_K_tl_K_l_mat), alpha_l)
    beta2 = dot(beta.T, beta)
    alpha_int_alpha = dot(dot(alpha_l.T, int_K_l_K_tl_K_l), alpha_l)
    E_m_l_C_tl_m_l = float(alpha_int_alpha - beta2)
    assert E_m_l_C_tl_m_l > 0

    ## Second term
    # E[m_l C_tl | x_s] =
    #    [ int int K_tl(x', x) K_l(x, x_s) p(x) p(x') dx dx' -
    #      ( int K_tl(x, x_s) p(x) dx) *
    #        inv(K_tl(x_s, x_s)) *
    #        int K_tl(x_s, x) K_l(x, x_s) p(x) dx
    #      )
    #    ] alpha_l(x_s)
    int_K_tl_K_l_vec = gaussint4(x_s, h_tl, w_tl, h_l, w_l, mu, cov)
    int_K_tl_vec = gaussint1(x_s, h_tl, w_tl, mu, cov)
    int_inv_int = dot(dot(int_K_tl_vec, inv_K_tl), int_K_tl_K_l_mat)
    E_m_l_C_tl = float(dot(int_K_tl_K_l_vec - int_inv_int, alpha_l))
    if E_m_l_C_tl <= 0:
        print "E[m_l C_tl] = %f" % E_m_l_C_tl
        assert False

    ## Third term
    # E[C_tl | x_s] =
    #    int int K_tl(x, x') p(x) p(x') dx dx' -
    #    ( int K_tl(x, x_s) p(x) dx *
    #      inv(K_tl(x_s, x_s)) *
    #      [int K_tl(x, x_s) p(x) dx]'
    #    )
    # Where eta is defined as:
    # eta(x_s) = inv(L_tl(x_s, x_s)) int K_tl(x_s, x) p(x) dx
    int_K_tl_scalar = gaussint5(d, h_tl, w_tl, mu, cov)
    int_inv_int_tl = dot(dot(int_K_tl_vec, inv_K_tl), int_K_tl_vec.T)
    E_C_tl = float(int_K_tl_scalar - int_inv_int_tl)
    assert E_C_tl > 0

    term1 = E_m_l_C_tl_m_l
    term2 = 2 * gamma * E_m_l_C_tl
    term3 = gamma ** 2 * E_C_tl
    V_Z = term1 + term2 + term3

    ##############################################################
    ## Variance correction

    zeta = dot(inv_K_tl, dK_tl_dw)
    dK_const1, dK_const2 = dtheta_consts(x_s, w_l, w_tl, mu, cov)

    term1 = np.zeros((1, 1))
    term2 = np.zeros((1, 1))
    for i in xrange(d):
            
        ## First term of nu
        int_K_tl_dK_l_mat = int_K_tl_K_l_mat * dK_const1[:, :, i]
        term1a = dot(dot(alpha_l.T, int_K_tl_dK_l_mat), alpha_tl)
        term1b = dot(dot(alpha_l.T, int_K_tl_K_l_mat, zeta, alpha_tl))
        term1 += term1a - term1b

        ## Second term of nu
        int_dK_tl_vec = int_K_tl_vec * dK_const2[None, :, i]
        term2a = dot(int_dK_tl_vec, alpha_tl)
        term2b = dot(dot(int_K_tl_vec, zeta, alpha_tl))
        term2 += term2a - term2b

    nu = term1 + gamma * term2
    V_Z_correction = float(dot(dot(nu, Cw), nu.T))
    V_Z += V_Z_correction
    assert V_Z > 0

    return V_Z
