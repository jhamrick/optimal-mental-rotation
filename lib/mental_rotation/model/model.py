import numpy as np
import pymc
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


def make_Xi(name, X):
    def Xa_logp(value):
        return prior(value)

    # we can't use the stochastic decorator because PyMC does weird
    # things with the system trace which screws up code coverage.
    Xi = pymc.Stochastic(
        logp=Xa_logp,
        doc='Stimulus %s' % name,
        name=name,
        parents={},
        random=None,
        trace=False,
        value=X,
        dtype=object,
        rseed=1.,
        observed=True,
        cache_depth=2,
        plot=False,
        verbose=0)

    return Xi


def make_F():
    F = pymc.Bernoulli("F", p=0.5, value=0, observed=False)
    return F


def make_R(R_mu, R_kappa):

    def R_logp(value, mu=R_mu, kappa=R_kappa):
        return pymc.distributions.von_mises_like(value, R_mu, R_kappa)

    def random(mu, kappa, size=1):
        return pymc.distributions.rvon_mises(mu, kappa, size=size)

    # we can't use the stochastic decorator because PyMC does weird
    # things with the system trace which screws up code coverage.
    R = pymc.Stochastic(
        logp=R_logp,
        doc='Rotation',
        name='R',
        parents={'mu': R_mu, 'kappa': R_kappa},
        random=random,
        trace=True,
        value=0,
        dtype=float,
        rseed=1.,
        observed=False,
        cache_depth=2,
        plot=False,
        verbose=0)

    return R


def make_Xr(Xa, R, F):
    @pymc.deterministic
    def Xr(Xa=Xa, R=R, F=F):
        Xr = Xa.copy()
        if F == 1:
            Stimulus2D._flip(Xr, np.array([0, 1]))
        Stimulus2D._rotate(Xr, np.degrees(R))
        return Xr
    return Xr


def make_log_S(Xb, Xr, S_sigma):
    @pymc.potential
    def log_S(Xb=Xb, Xr=Xr):
        return log_similarity(Xb, Xr, S_sigma)
    return log_S
