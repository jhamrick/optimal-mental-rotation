import numpy as np
import scipy

from .. import Stimulus2D
from .model_c import log_prior, log_similarity, log_const


def memoprop(f):
    """
    Memoized property.

    When the property is accessed for the first time, the return value
    is stored and that value is given on subsequent calls. The memoized
    value can be cleared by calling 'del prop', where prop is the name
    of the property.

    """
    fname = f.__name__

    def fget(self):
        if fname not in self._memoized:
            self._memoized[fname] = f(self)
        return self._memoized[fname]

    def fdel(self):
        del self._memoized[fname]

    prop = property(fget=fget, fdel=fdel, doc=f.__doc__)
    return prop


def slow_log_prior(X):
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


def slow_log_similarity(X0, X1, S_sigma):
    """Computes the similarity between sets of vertices `X0` and `X1`."""
    # number of points and number of dimensions
    n, D = X0.shape
    # covariance matrix
    Sigma = np.eye(D) * S_sigma
    invSigma = np.eye(D) * (1. / S_sigma)
    # constants
    Z0 = D * np.log(2 * np.pi)
    Z1 = np.linalg.slogdet(Sigma)[1]
    # iterate through all permutations of the vertices -- but if
    # two vertices are connected, they are next to each other in
    # the list (or on the ends), so we really only need to cycle
    # through 2n orderings (once for the original ordering, and
    # once for the reverse)
    e = np.empty(2*n)
    for i in xrange(n):
        idx = np.arange(i, i+n) % n
        d = X0 - X1[idx]
        e[i] = 0
        for j in xrange(n):
            e[i] += -0.5 * (np.dot(d[j], np.dot(invSigma, d[j])) + Z0 + Z1)
    for i in xrange(n):
        idx = np.arange(i, i+n)[::-1] % n
        d = X0 - X1[idx]
        e[i+n] = 0
        for j in xrange(n):
            e[i+n] += -0.5 * (np.dot(d[j], np.dot(invSigma, d[j])) + Z0 + Z1)
    # overall similarity, marginalizing out order
    log_S = np.log(np.sum(np.exp(e - np.log(n))))
    return log_S


class Variable(object):

    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
        self.children = []
        self._memoized = {}

        for p in self.parents:
            p.children.append(self)

    def clear(self):
        self._memoized = {}
        for child in self.children:
            child.clear()


class Xi(Variable):
    """Shape"""

    def __init__(self, name, value):
        super(Xi, self).__init__(name, [])

        self.observed = True
        self._value = value

    @property
    def value(self):
        return self._value

    @memoprop
    def logp(self):
        return log_prior(self.value)


class F(Variable):
    """Flipped"""

    def __init__(self):
        super(F, self).__init__("F", [])

        self.p = 0.5
        self.observed = False
        self._value = 0

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
        self.clear()
        
    @memoprop
    def logp(self):
        return np.log(self.p * self.value + (1 - self.p) * (1 - self.value))


class R(Variable):
    """Rotation"""

    def __init__(self, mu, kappa):
        super(R, self).__init__("R", [])

        self.mu = mu
        self.kappa = kappa
        self.observed = False
        self._value = 0

        self._C = -np.log(2 * np.pi * scipy.special.iv(0, self.kappa))

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        val = val % (2 * np.pi)
        if val > np.pi:
            val -= 2 * np.pi
        self._value = val
        self.clear()
        
    @memoprop
    def logp(self):
        logp = self._C + (self.kappa * np.cos(self.value - self.mu))
        return logp
        

class Xr(Variable):
    """Mental image"""
    
    def __init__(self, Xa, R, F):
        super(Xr, self).__init__("Xr", [Xa, R, F])

        self.Xa = Xa
        self.R = R
        self.F = F

    @memoprop
    def value(self):
        Xr = self.Xa.value.copy()
        if self.F.value == 1:
            Stimulus2D._flip(Xr, np.array([0, 1]))
        Stimulus2D._rotate(Xr, np.degrees(self.R.value))
        return Xr


class log_S(Variable):
    """Log similarity"""

    def __init__(self, Xb, Xr, sigma):
        super(log_S, self).__init__("log_S", [Xb, Xr])

        self.Xb = Xb
        self.Xr = Xr
        self.sigma = sigma

    @memoprop
    def logp(self):
        return log_similarity(self.Xb.value, self.Xr.value, self.sigma)


class log_dZ_dR(Variable):

    def __init__(self, log_S, R, F):
        super(log_dZ_dR, self).__init__(
            "log_dZ_dR", [log_S, R, F])

        self.log_S = log_S
        self.R = R
        self.F = F
        
    @memoprop
    def logp(self):
        p_log_S = self.log_S.logp
        p_R = self.R.logp
        p_F = self.F.logp
        return p_log_S + p_R + p_F
