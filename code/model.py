import numpy as np

from snippets.stats import GP
from snippets.stats import circular_gaussian_kernel as make_kernel


def similarity(I0, I1, sf=1):
    """Computes the similarity between images I0 and I1"""
    # C = log(1. / sqrt(2 * pi * sf**2))
    diff = np.exp(np.sum(-0.5 * (I0 - I1)**2 / (sf**2))) + 1
    return diff


def fit_likelihood(R, Sr, iix, cix):
    """

    References
    ----------

    Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
        Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
        Model Evidence Using Bayesian Quadrature. *Advances in Neural
        Information Processing Systems*, 25.

    """

    # input points for the GPs over S and log S
    xi = R[iix].copy()
    yi = Sr[iix].copy()

    # GPR for S
    mu_S, cov_S = GP(make_kernel(2, np.pi/4.), xi, yi, R)
    # GPR for log S
    mu_logS, cov_logS = GP(make_kernel(np.log(2), np.pi/4.), xi, np.log(yi), R)

    # "candidate" points
    delta = mu_logS - np.log(mu_S)
    xc = R[cix].copy()
    yc = delta[cix].copy()

    # GPR for Delta
    mu_Dc, cov_Dc = GP(make_kernel(0.01, np.pi/4.), xc, yc, R)

    return (delta, xi, yi, xc, yc,
            mu_S, cov_S, mu_logS, cov_logS, mu_Dc, cov_Dc)
