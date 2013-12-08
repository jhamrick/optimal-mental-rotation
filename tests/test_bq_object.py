import numpy as np
import scipy.stats
import pytest

from mental_rotation import config
from mental_rotation.extra import BQ
import mental_rotation.extra.bq_c as bq_c
from . import util

import logging
logger = logging.getLogger("mental_rotation.extra.bq")
logger.setLevel("DEBUG")


gamma = config.getfloat("bq", "gamma")
ntry = config.getint("bq", "ntry")
n_candidate = config.getint("bq", "n_candidate")
R_mean = config.getfloat("model", "R_mu")
R_var = 1. / config.getfloat("model", "R_kappa")


def make_1d_gaussian(x=None, seed=True, n=20):
    if seed:
        util.seed()
    if x is None:
        x = np.random.uniform(-5, 5, n)
    y = scipy.stats.norm.pdf(x, 0, 1)
    return x, y


def make_bq(seed=True, n=20):
    x, y = make_1d_gaussian(seed=seed, n=n)
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    return bq


def test_improve_covariance_conditioning():
    bq = make_bq()
    bq.fit()

    K_l = bq.gp_S.Kxx
    bq_c.improve_covariance_conditioning(K_l)
    assert (K_l == bq.gp_S.Kxx).all()
    assert K_l is bq.gp_S.Kxx

    K_tl = bq.gp_log_S.Kxx
    bq_c.improve_covariance_conditioning(K_tl)
    assert (K_tl == bq.gp_log_S.Kxx).all()
    assert K_tl is bq.gp_log_S.Kxx

    K_del = bq.gp_Dc.Kxx
    bq_c.improve_covariance_conditioning(K_del)
    assert (K_del == bq.gp_Dc.Kxx).all()
    assert K_del is bq.gp_Dc.Kxx


def test_init():
    x, y = make_1d_gaussian()
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    assert (x == bq.R).all()
    assert (y == bq.S).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], gamma, ntry, n_candidate, R_mean, R_var, s=0)


def test_log_transform():
    x, y = make_1d_gaussian()
    log_y = np.log((y / float(gamma)) + 1)
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    assert np.allclose(bq.log_S, log_y)


def test_choose_candidates():
    bq = make_bq()
    bq._fit_S()
    Rc = bq._choose_candidates()
    assert Rc.ndim == 1
    assert Rc.size >= bq.R.size

    diff = np.abs(Rc[:, None] - bq.R[None])
    assert ((diff > 1e-4) | (diff == 0)).all()


def test_S_mean():
    util.seed()

    for i in xrange(10):
        bq = make_bq(seed=False)
        bq.fit()

        x, y = make_1d_gaussian(np.linspace(-5, 5, 100))
        diff = bq.S_mean(x) - y
        assert np.percentile(np.abs(diff), 50) < 1e-3


# def test_S_cov():
#     raise NotImplementedError


def test_mvn_logpdf():
    util.seed()
    x = np.random.uniform(-5, 5, 20)
    y = scipy.stats.norm.pdf(x, R_mean, np.sqrt(R_var))
    pdf = np.empty_like(y)
    mu = np.array([R_mean])
    cov = np.array([[R_var]])
    bq_c.mvn_logpdf(pdf, x[:, None], mu, cov)
    assert np.allclose(np.log(y), pdf)


def test_int_K():
    bq = make_bq()
    bq.fit()

    xo = np.linspace(-10, 10, 10000)

    Kxxo = bq.gp_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    int_K = np.empty(bq.R.shape[0])
    bq_c.int_K(
        int_K, bq.R[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(int_K, approx_int)

    Kxxo = bq.gp_log_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    bq_c.int_K(
        int_K, bq.R[:, None],
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(int_K, approx_int)


def test_int_K1_K2():
    bq = make_bq(n=10)
    bq.fit()

    xo = np.linspace(-10, 10, 10000)

    Kx1xo = bq.gp_S.Kxxo(xo)
    Kx2xo = bq.gp_log_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(Kx1xo[:, None] * Kx2xo[None, :] * p_xo, xo)

    int_K1_K2 = np.empty((bq.R.shape[0], bq.R.shape[0]))
    bq_c.int_K1_K2(
        int_K1_K2, bq.gp_S.x[:, None], bq.gp_log_S.x[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)

    assert np.allclose(int_K1_K2, approx_int, rtol=1e-5, atol=1e-4)


# def test_gaussint3():
#     raise NotImplementedError
#     assert mat.shape == (n, n)


# def test_gaussint4():
#     raise NotImplementedError
#     assert vec.shape == (1, n)


# def test_gaussint5():
#     raise NotImplementedError


# def test_dtheta_consts():
#     raise NotImplementedError


# def test_Z_mean():
#     raise NotImplementedError


# def test_Z_var():
#     raise NotImplementedError


# def test_dm_dw():
#     raise NotImplementedError


# def test_Cw():
#     raise NotImplementedError


# def test_expected_uncertainty_evidence():
#     raise NotImplementedError
