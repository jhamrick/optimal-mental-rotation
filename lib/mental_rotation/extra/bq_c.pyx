from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport exp, log, fmax, copysign, fabs, M_PI
from cpython cimport bool

cdef inv = np.linalg.inv
cdef slogdet = np.linalg.slogdet
cdef dot = np.dot

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps


def mvn_logpdf(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=2] C, bool use_Z, DTYPE_t Z):
    """Computes the logpdf for a multivariate normal distribution:

    out[i] = N(x_i | m, C)
           = -0.5*log(2*pi)*d - 0.5*(x_i-m)*C^-1*(x_i-m) + Z

    where Z is either provided (i.e. `use_Z` is True) or Z = -0.5*log(|C|)

    """
    cdef np.ndarray[DTYPE_t, ndim=2] Ci
    cdef int n, d, i, j, k
    cdef DTYPE_t t0, t1

    n = x.shape[0]
    d = x.shape[1]
    Ci = inv(C)

    if not use_Z:
        Z = -0.5 * slogdet(C)[1]        

    t0 = log(2 * M_PI) * (-d / 2.) + Z

    for i in xrange(n):
        t1 = 0
        for j in xrange(d):
            for k in xrange(d):
                t1 += (x[i] - m[j]) * Ci[j, k] * (x[i] - m[k])
        out[i] = t0 - 0.5 * t1


def improve_covariance_conditioning(np.ndarray[DTYPE_t, ndim=2] M):
    cdef DTYPE_t sqd_jitters = fmax(EPS, np.max(M)) * 1e-4
    cdef int n, i
    n = M.shape[0]
    for i in xrange(n):
        M[i, i] += sqd_jitters


def gaussint1(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h, DTYPE_t w, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int K(x', x) N(x' | mu, cov) dx'

    where K is a Gaussian kernel matrix parameterized by `h` and `w`.

    The result is:

    out[i] = h^2 N(x_i | mu, W + cov)

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W
    cdef int n, d, i, j
    cdef DTYPE_t h_2

    n = x.shape[0]
    d = x.shape[1]
    W = np.empty((d, d), dtype=DTYPE)
    h_2 = h ** 2

    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W[i, j] = cov[i, j] + w ** 2
            else:
                W[i, j] = cov[i, j]

    mvn_logpdf(out, x, mu, W, False, 0)
    for i in xrange(n):
        out[i] = h_2 * exp(out[i])


def gaussint2(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2, DTYPE_t h1, DTYPE_t w1, DTYPE_t h2, DTYPE_t w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int K_1(x1, x') K_2(x', x2) N(x' | mu, cov) dx'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^2 h2^2 N([x1_i, x2_j] | [mu, mu], [W1 + cov, cov; cov, W2 + cov])

    """
    
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef np.ndarray[DTYPE_t, ndim=1] m
    cdef np.ndarray[DTYPE_t, ndim=2] C
    cdef int n1, n2, d, i, j, k
    cdef DTYPE_t ha, hb

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]

    x = np.empty(2 * d, dtype=DTYPE)
    m = np.empty(2 * d, dtype=DTYPE)
    C = np.empty((2 * d, 2 * d), dtype=DTYPE)

    h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)

    # compute concatenated means [mu, mu]
    for i in xrange(d):
        m[i] = mu[k]
        m[i + d] = mu[k]

    # compute concatenated covariances [W1 + cov, cov; cov; W2 + cov]
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w1 + cov[i, j]
                C[i + d, j + d] = w2 + cov[i, j]
            else:
                C[i, j] = cov[i, j]
                C[i + d, j + d] = cov[i, j]

            C[i, j + d] = cov[i, j]
            C[i + d, j] = cov[i, j]

    # compute concatenated x
    for i in xrange(n1):
        for j in xrange(n2):
            for k in xrange(d):
                x[i, j, k] = x1[i, k]
                x[i, j, k + d] = x2[j, k]

    # compute pdf
    for i in xrange(n1):
        mvn_logpdf(out[i], x[i], m, C, False, 0)
        for j in xrange(n2):
            out[i, j] = h1_2_h2_2 * exp(out[i, j])


def gaussint3(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h1, DTYPE_t w1, DTYPE_t h2, DTYPE_t w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x, x1') K_2(x1', x2') K_1(x2', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^4 h2^2 |G|^-1 N(x_i | mu, W1 + cov) N(x_j | mu, W1 + cov) N(x_i | x_j, G^-1 (W2 + 2*cov - 2*G*cov) G^-1)

    where G = cov(W1 + cov)^-1

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W1_cov
    cdef np.ndarray[DTYPE_t, ndim=2] G
    cdef np.ndarray[DTYPE_t, ndim=2] Gi
    cdef np.ndarray[DTYPE_t, ndim=2] GWG
    cdef np.ndarray[DTYPE_t, ndim=1] N1
    cdef np.ndarray[DTYPE_t, ndim=2] N2
    cdef int n, d, i, j
    cdef DTYPE_t h1_4, h2_2, Gdeti

    n = x.shape[0]
    d = x.shape[1]

    h1_4_h2_2 = (h1 ** 4) * (h2 ** 2)

    # compute W1 + cov
    W1_cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            W1_cov[i, j] = cov[i, j] + w1

    # compute G = cov*(W1 + cov)^-1
    G = dot(cov, inv(W1_cov))
    Gi = inv(G)
    Gcov = dot(G, cov)
    Gdeti = -slogdet(G)[1]

    # compute G^-1 (W2 + 2*cov - 2*G*cov) G^-1
    GWG = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            GWG[i, j] = w2 + 2*cov[i, j] - 2*Gcov[i, j]
    GWG[:] = dot(dot(Gi, GWG), Gi)

    # compute N(x | mu, W1 + cov)
    N1 = np.empty(n, dtype=DTYPE)
    mvn_logpdf(N1, x, mu, W1_cov, False, 0)

    # compute N(x_i | x_j, G^-1 (W2 + 2*cov - 2*G*cov) G^-1)
    N2 = np.empty((n, n), dtype=DTYPE)
    for j in xrange(n):
        mvn_logpdf(N2[i], x, x[j], GWG, False, 0)

    # put it all together
    for i in xrange(n):
        for j in xrange(n):
            out[i, j] = h1_4_h2_2 * exp(Gdeti + N1[i] + N1[j] + N2[i, j])


def gaussint4(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h1, DTYPE_t w1, DTYPE_t h2, DTYPE_t w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x2', x1') K_2(x1', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i] = h1^2 h2^2 N(0 | 0, W1 + 2*cov) N(x_i | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W1_2cov
    cdef np.ndarray[DTYPE_t, ndim=2] C
    cdef np.ndarray[DTYPE_t, ndim=1] N1
    cdef np.ndarray[DTYPE_t, ndim=2] N2
    cdef int n, d, i, j
    cdef DTYPE_t h1_2, h2_2

    n = x.shape[0]
    d = x.shape[1]

    h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)

    # compute W1 + 2*cov
    W1_2cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            W1_2cov[i, j] = 2*cov[i, j] + w1

    # compute N(0 | 0, W1 + 2*cov)
    N1 = np.empty(1, dtype=DTYPE)
    zx = np.zeros((1, d), dtype=DTYPE)
    zm = np.zeros(d, dtype=DTYPE)
    mvn_logpdf(N1, zx, zm, W1_2cov, False, 0)

    # compute W2 + cov - cov*(W1 + 2*cov)^-1*cov
    C = dot(dot(cov, W1_2cov), inv(cov))
    for i in xrange(d):
        for j in xrange(d):
            C[i, j] = w2 + cov[i, j] - C[i, j]

    # compute N(x | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)
    N2 = np.empty(1, dtype=DTYPE)
    mvn_logpdf(N2, x, mu, C, False, 0)

    for i in xrange(n):
        out[i] = h1_2_h2_2 * exp(N1[i] + N2[i])

# def gaussint5(d, h, w, mu, cov):
#     h2 = h ** 2
#     W = (np.array(w) * np.eye(d)) ** 2
#     const = float(h2 * np.exp(mvn_logpdf(
#         np.zeros((1, d)),
#         np.zeros(d),
#         W + 2*cov)))
#     return const

# def dtheta_consts(x, w1, w2, mu, L):
#     n, d = x.shape
#     I = np.eye(d)
#     Wl = (w1 ** 2) * I
#     Wtl = (w2 ** 2) * I
#     iWtl = inv(Wtl)

#     A = dot(L, inv(Wtl + L))
#     B = L - dot(A, L)
#     C = dot(B, inv(B + Wl))
#     CA = dot(C, A)
#     BCB = B - dot(C, B)

#     xsubmu = x - mu
#     mat_const = np.empty((n, n, d))
#     vec_const = np.empty((n, d))
#     c = -0.5*iWtl

#     m1 = dot(A - I, xsubmu.T).T
#     m2a = dot(A - CA - I, xsubmu.T).T
#     m2b = dot(C, xsubmu.T).T
    
#     for i in xrange(n):
#         vec_const[i] = np.diag(
#             c * (I + dot(B - dot(m1[i], m1[i].T), iWtl)))

#         for j in xrange(n):
#             m2 = m2a[i] + m2b[j]
#             mat_const[i, j] = np.diag(
#                 c * (I + dot(BCB - dot(m2, m2.T), iWtl)))
            
#     return mat_const, vec_const

# def Z_mean(x_s, x_sc, alpha_l, alpha_del, h_s, w_s, h_dc, w_dc, mu, cov, gamma):

#     ns, d = x_s.shape
#     nc, d = x_sc.shape

#     cdef np.ndarray[DTYPE_t, ndim=1] int_K_l = np.empty(ns, dtype=DTYPE)
#     cdef np.ndarray[DTYPE_t, ndim=1] int_K_del = np.empty(nc, dtype=DTYPE)

#     ## First term
#     # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
#     gaussint1(int_K_l, x_s, h_s, w_s, mu, cov)
#     E_m_l = float(dot(int_K_l, alpha_l))
#     assert E_m_l > 0

#     ## Second term
#     # E[m_l*m_del | x_s, x_c] = alpha_del(x_sc)' *
#     #     int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
#     #     alpha_l(x_s)
#     int_K_del_K_l = gaussint2(
#         x_sc, x_s, h_dc, w_dc, h_s, w_s, mu, cov)
#     E_m_l_m_del = float(dot(dot(
#         alpha_del.T, int_K_del_K_l), alpha_l))
    
#     ## Third term
#     # E[m_del | x_sc] = (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
#     gaussint1(int_K_del, x_sc, h_dc, w_dc, mu, cov)
#     E_m_del = float(dot(int_K_del, alpha_del))
    
#     # put the three terms together
#     m_Z = E_m_l + E_m_l_m_del + gamma * E_m_del

#     return m_Z


# def Z_var(x_s, alpha_l, alpha_tl, inv_L_tl, inv_K_tl, dK_tl_dw, Cw, h_l, w_l, h_tl, w_tl, mu, cov, gamma):
#     ns, d = x_s.shape

#     cdef np.ndarray[DTYPE_t, ndim=1] int_K_tl_vec = np.empty(ns, dtype=DTYPE)

#     ## First term
#     # E[m_l C_tl m_l | x_s] = alpha_l(x_s)' *
#     #    int int K_l(x_s, x) K_tl(x, x') K_l(x', x_s) p(x) p(x') dx dx' *
#     #    alpha_l(x_s) - beta(x_s)'beta(x_s)
#     # Where beta is defined as:
#     # beta(x_s) = inv(L_tl(x_s, x_s)) *
#     #    int K_tl(x_s, x) K_l(x, x_s) p(x) dx *
#     #    alpha_l(x_s)
#     int_K_l_K_tl_K_l = gaussint3(
#         x_s, h_l, w_l, h_tl, w_tl, mu, cov)
#     int_K_tl_K_l_mat = gaussint2(
#         x_s, x_s, h_tl, w_tl, h_l, w_l, mu, cov)
#     beta = dot(dot(inv_L_tl, int_K_tl_K_l_mat), alpha_l)
#     beta2 = dot(beta.T, beta)
#     alpha_int_alpha = dot(dot(alpha_l.T, int_K_l_K_tl_K_l), alpha_l)
#     E_m_l_C_tl_m_l = float(alpha_int_alpha - beta2)
#     assert E_m_l_C_tl_m_l > 0

#     ## Second term
#     # E[m_l C_tl | x_s] =
#     #    [ int int K_tl(x', x) K_l(x, x_s) p(x) p(x') dx dx' -
#     #      ( int K_tl(x, x_s) p(x) dx) *
#     #        inv(K_tl(x_s, x_s)) *
#     #        int K_tl(x_s, x) K_l(x, x_s) p(x) dx
#     #      )
#     #    ] alpha_l(x_s)
#     int_K_tl_K_l_vec = gaussint4(x_s, h_tl, w_tl, h_l, w_l, mu, cov)
#     gaussint1(int_K_tl_vec, x_s, h_tl, w_tl, mu, cov)
#     int_inv_int = dot(dot(int_K_tl_vec, inv_K_tl), int_K_tl_K_l_mat)
#     E_m_l_C_tl = float(dot(int_K_tl_K_l_vec - int_inv_int, alpha_l))
#     if E_m_l_C_tl <= 0:
#         print "E[m_l C_tl] = %f" % E_m_l_C_tl
#         assert False

#     ## Third term
#     # E[C_tl | x_s] =
#     #    int int K_tl(x, x') p(x) p(x') dx dx' -
#     #    ( int K_tl(x, x_s) p(x) dx *
#     #      inv(K_tl(x_s, x_s)) *
#     #      [int K_tl(x, x_s) p(x) dx]'
#     #    )
#     # Where eta is defined as:
#     # eta(x_s) = inv(L_tl(x_s, x_s)) int K_tl(x_s, x) p(x) dx
#     int_K_tl_scalar = gaussint5(d, h_tl, w_tl, mu, cov)
#     int_inv_int_tl = dot(dot(int_K_tl_vec, inv_K_tl), int_K_tl_vec.T)
#     E_C_tl = float(int_K_tl_scalar - int_inv_int_tl)
#     assert E_C_tl > 0

#     term1 = E_m_l_C_tl_m_l
#     term2 = 2 * gamma * E_m_l_C_tl
#     term3 = gamma ** 2 * E_C_tl
#     V_Z = term1 + term2 + term3

#     ##############################################################
#     ## Variance correction

#     zeta = dot(inv_K_tl, dK_tl_dw)
#     dK_const1, dK_const2 = dtheta_consts(x_s, w_l, w_tl, mu, cov)

#     term1 = np.zeros((1, 1))
#     term2 = np.zeros((1, 1))
#     for i in xrange(d):
            
#         ## First term of nu
#         int_K_tl_dK_l_mat = int_K_tl_K_l_mat * dK_const1[:, :, i]
#         term1a = dot(dot(alpha_l.T, int_K_tl_dK_l_mat), alpha_tl)
#         term1b = dot(dot(alpha_l.T, int_K_tl_K_l_mat, zeta, alpha_tl))
#         term1 += term1a - term1b

#         ## Second term of nu
#         int_dK_tl_vec = int_K_tl_vec * dK_const2[None, :, i]
#         term2a = dot(int_dK_tl_vec, alpha_tl)
#         term2b = dot(dot(int_K_tl_vec, zeta, alpha_tl))
#         term2 += term2a - term2b

#     nu = term1 + gamma * term2
#     V_Z_correction = float(dot(dot(nu, Cw), nu.T))
#     V_Z += V_Z_correction
#     assert V_Z > 0

#     return V_Z
