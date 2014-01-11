import numpy as np
import scipy

from .. import Stimulus2D


def prior(X):
    # the beginning is the same as the end, so ignore the last vertex
    n = X.shape[0] - 1
    # n points picked at random angles around the circle
    log_pangle = -np.log(2*np.pi) * n
    # random radii between 0 and 1
    radius = 1
    log_pradius = -np.log(radius) * n
    # number of possible permutations of the points
    log_pperm = np.log(scipy.misc.factorial(n))
    # put it all together
    logp = log_pperm + log_pangle + log_pradius
    return logp


def log_similarity(X0, X1, S_sigma):
    """Computes the similarity between sets of vertices `X0` and `X1`."""
    # number of points and number of dimensions
    n, D = X0.shape
    # covariance matrix
    Sigma = np.eye(D) * S_sigma
    invSigma = np.eye(D) * (1. / S_sigma)
    # iterate through all permutations of the vertices -- but if
    # two vertices are connected, they are next to each other in
    # the list (or on the ends), so we really only need to cycle
    # through 2n orderings (once for the original ordering, and
    # once for the reverse)
    e = np.empty(2*n)
    for i in xrange(n):
        idx = np.arange(i, i+n) % n
        d = X0 - X1[idx]
        e[i] = -0.5 * np.sum(np.dot(d, invSigma) * d)
    for i in xrange(n):
        idx = np.arange(i, i+n)[::-1] % n
        d = X0 - X1[idx]
        e[i+n] = -0.5 * np.sum(np.dot(d, invSigma) * d)
    # constants
    Z0 = (D / 2.) * np.log(2 * np.pi)
    Z1 = 0.5 * np.linalg.slogdet(Sigma)[1]
    # overall similarity, marginalizing out order
    log_S = np.log(np.sum(np.exp(e + Z0 + Z1 - np.log(n))))
    return log_S


class Xi(object):
    """Shape"""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.observed = True

    @property
    def logp(self):
        return prior(self.value)


class F(object):
    """Flipped"""

    def __init__(self):
        self.name = "F"
        self.p = 0.5
        self.value = 0
        self.observed = False

    @property
    def logp(self):
        return np.log(self.p * self.value + (1 - self.p) * (1 - self.value))


class R(object):
    """Rotation"""

    def __init__(self, mu, kappa):
        self.name = "R"
        self.mu = mu
        self.kappa = kappa
        self.value= 0
        self.observed = False

        self._C = -np.log(2 * np.pi * scipy.special.iv(0, self.kappa))

    @property
    def logp(self):
        logp = self._C + (self.kappa * np.cos(self.value - self.mu))
        return logp
        

class Xr(object):
    """Mental image"""
    
    def __init__(self, Xa, R, F):
        self.name = "Xr"
        self.Xa = Xa
        self.R = R
        self.F = F

    @property
    def value(self):
        Xr = self.Xa.value.copy()
        if self.F.value == 1:
            Stimulus2D._flip(Xr, np.array([0, 1]))
        Stimulus2D._rotate(Xr, np.degrees(self.R.value))
        return Xr


class log_S(object):
    """Log similarity"""

    def __init__(self, Xb, Xr, sigma):
        self.name = "log_S"
        self.Xb = Xb
        self.Xr = Xr
        self.sigma = sigma

    @property
    def logp(self):
        return log_similarity(self.Xb.value, self.Xr.value, self.sigma)


class log_dZ_dR(object):
    def __init__(self, log_S, R, F):
        self.name = "log_dZ_dR"
        self.log_S = log_S
        self.R = R
        self.F = F
        
    @property
    def logp(self):
        p_log_S = self.log_S.logp
        p_R = self.R.logp
        p_F = self.F.logp
        return p_log_S + p_R + p_F
