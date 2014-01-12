# cython: embedsignature=True

import numpy as np
import scipy.stats
import scipy.linalg
from numpy.linalg import LinAlgError
from warnings import warn
from numpy import float64, int32
from numpy import empty

######################################################################

from libc.math cimport exp, log, fmax, fabs, M_PI, cos, INFINITY
from numpy cimport float64_t

cimport bayesian_quadrature.linalg_c as la
cimport bayesian_quadrature.gauss_c as ga

######################################################################

cdef float64_t MIN = log(np.exp2(float64(np.finfo(float64).minexp + 4)))
cdef float64_t EPS = np.finfo(float64).eps
cdef float64_t NAN = np.nan

######################################################################


cpdef float64_t log_factorial(long n):
    cdef float64_t fac = 0.0
    cdef int i
    for i in xrange(2, n+1):
        fac += log(i)
    return fac


def log_prior(float64_t[:, ::1] X):
    # the beginning is the same as the end, so ignore the last vertex
    cdef int n = X.shape[0] - 1

    # n points picked at random angles around the circle
    cdef float64_t log_pangle = -log(2 * M_PI) * n

    # technically, we should have a term here for log(p(radius)), but
    # because the radius is 1, this is zero, so we don't bother
    # actually computing it
    #     log_pradius = -log(radius) * n

    # number of possible permutations of the points
    cdef float64_t log_pperm = log_factorial(n)

    # put it all together
    cdef float64_t logp = log_pperm + log_pangle

    return logp
    

def log_similarity(float64_t[:, ::1] X0, float64_t[:, ::1] X1, float64_t S_sigma):
    """Computes the similarity between sets of vertices `X0` and `X1`."""
    # number of points and number of dimensions
    cdef int n = X0.shape[0]
    cdef int D = X0.shape[1]

    cdef float64_t[::1, :] S = np.empty((D, D), dtype=float64, order='F')
    cdef float64_t logp, log_S
    cdef int i, j, k, idx

    # covariance matrix
    S[:, :] = 0
    for i in xrange(D):
        S[i, i] = S_sigma
    la.cho_factor(S, S)
    logdet = la.logdet(S)

    # iterate through all permutations of the vertices -- but if
    # two vertices are connected, they are next to each other in
    # the list (or on the ends), so we really only need to cycle
    # through 2n orderings (once for the original ordering, and
    # once for the reverse)
    log_S = 0
    for i in xrange(n):
        logp = 0
        for j in xrange(n):
            logp += ga.mvn_logpdf(X0[j], X1[(j + i) % n], S, logdet)
        log_S += exp(logp - log(n))

        logp = 0
        for j in xrange(n):
            logp += ga.mvn_logpdf(X0[j], X1[(n - j + i - 1) % n], S, logdet)
        log_S += exp(logp - log(n))

    log_S = log(log_S)
    return log_S
