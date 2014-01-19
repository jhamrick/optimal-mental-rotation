# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True

from numpy import float64, int32
from numpy import empty

######################################################################

from libc.math cimport exp, log, M_PI, fmod
from numpy cimport float64_t

cimport bayesian_quadrature.linalg_c as la
cimport bayesian_quadrature.gauss_c as ga

######################################################################


cdef float64_t _log_factorial(long n):
    cdef float64_t fac = 0.0
    cdef int i
    for i in xrange(2, n+1):
        fac += log(i)
    return fac


def log_factorial(long n):
    return _log_factorial(n)


cpdef float64_t log_const(long n, long d, float64_t S_sigma):
    cdef float64_t const = -0.5 * (log(S_sigma) + log(2 * M_PI)) * n * d - log(2 * n)
    return const


cpdef float64_t log_prior(float64_t[:, ::1] X):
    # the beginning is the same as the end, so ignore the last vertex
    cdef int n = X.shape[0] - 1

    # n points picked at random angles around the circle
    cdef float64_t log_pangle = -log(2 * M_PI) * n

    # technically, we should have a term here for log(p(radius)), but
    # because the radius is 1, this is zero, so we don't bother
    # actually computing it
    #     log_pradius = -log(radius) * n

    # number of possible permutations of the points
    cdef float64_t log_pperm = _log_factorial(n)

    # put it all together
    cdef float64_t logp = log_pperm + log_pangle

    return logp
    

cpdef float64_t log_similarity(float64_t[:, ::1] X0, float64_t[:, ::1] X1, float64_t S_sigma):
    """Computes the similarity between sets of vertices `X0` and `X1`."""
    # number of points and number of dimensions
    cdef int n = X0.shape[0]
    cdef int D = X0.shape[1]

    cdef float64_t[::1, :] S = empty((D, D), dtype=float64, order='F')
    cdef float64_t[::1, :] logp = empty((n, n), dtype=float64, order='F')
    cdef float64_t total1, total2, log_S, logn
    cdef int i, j, idx

    # covariance matrix
    S[:, :] = 0
    for i in xrange(D):
        S[i, i] = S_sigma
    la.cho_factor(S, S)
    logdet = la.logdet(S)

    for i in xrange(n):
        for j in xrange(n):
            logp[i, j] = ga.mvn_logpdf(X0[i], X1[j], S, logdet)

    # iterate through all permutations of the vertices -- but if
    # two vertices are connected, they are next to each other in
    # the list (or on the ends), so we really only need to cycle
    # through 2n orderings (once for the original ordering, and
    # once for the reverse)
    log_S = 0
    for i in xrange(n):
        total1 = 0
        total2 = 0
        for j in xrange(n):
            idx = <int>fmod(j + i, n)
            total1 += logp[j, idx]
            idx = <int>fmod(n - j + i - 1, n)
            total2 += logp[j, idx]
        log_S += exp(total1) + exp(total2)

    log_S = log(log_S / (2.0 * n))
    return log_S
