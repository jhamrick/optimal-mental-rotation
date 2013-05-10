import numpy as np

from model_naive import NaiveModel as Naive
from model_vm import VonMisesModel as VonMises
from model_bq import BayesianQuadratureModel as BayesianQuadrature


def log_prior_X(X):
    # the beginning is the same as the end, so ignore the last vertex
    x = X[:-1]
    # number of points and number of dimensions
    n, D = x.shape
    assert D == 2
    # n points picked at random angles around the circle
    log_pangle = -np.log(2*np.pi) * n
    # one point has radius 1, the rest have random radii (this term
    # doesn't actually matter in practice, but leaving it in to be
    # explicit)
    radius = 1
    log_pradius = np.log(radius) * (n-1)
    # put it all together
    p_X = log_pangle + log_pradius
    return p_X


def similarity(X0, X1, sf=1):
    """Computes the similarity between sets of vertices `X0` and `X1`."""
    # the beginning is the same as the end, so ignore the last vertex
    x0 = X0[:-1]
    x1 = X1[:-1]
    # number of points and number of dimensions
    n, D = x0.shape
    # covariance matrix
    Sigma = np.eye(D) * sf
    invSigma = np.eye(D) * (1. / sf)
    # iterate through all permutations of the vertices -- but if two
    # vertices are connected, they are next to each other in the list
    # (or on the ends), so we really only need to cycle through n
    # orderings
    e = np.empty(n)
    for i in xrange(n):
        idx = np.arange(i, i+n) % n
        d = x0 - x1[idx]
        e[i] = -0.5 * np.sum(np.dot(d, invSigma) * d)
    # constants
    Z0 = (D / 2.) * np.log(2 * np.pi)
    Z1 = 0.5 * np.linalg.slogdet(Sigma)[1]
    # overall similarity, marginalizing out order
    S = np.sum(np.exp(e + Z0 + Z1 - np.log(n)))
    return S
