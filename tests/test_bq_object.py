import numpy as np
import scipy.stats
import pytest

from mental_rotation.extra import BQ
from . import util

import logging
logger = logging.getLogger("mental_rotation.extra.bq")
logger.setLevel("DEBUG")


gamma = 1
ntry = 10
n_candidate = 10


def make_1d_gaussian(x=None, seed=True):
    if seed:
        util.seed()
    if x is None:
        x = np.random.uniform(-5, 5, 20)
    y = scipy.stats.norm.pdf(x, 0, 1)
    return x, y


def make_bq(seed=True):
    x, y = make_1d_gaussian(seed=seed)
    bq = BQ(x, y, gamma, ntry, n_candidate, s=0)
    return bq


def test_improve_covariance_conditioning():
    bq = make_bq()
    bq.fit()

    K_l = bq.gp_S.Kxx
    bq._improve_covariance_conditioning(K_l)
    assert (K_l == bq.gp_S.Kxx).all()
    assert K_l is bq.gp_S.Kxx

    K_tl = bq.gp_log_S.Kxx
    bq._improve_covariance_conditioning(K_tl)
    assert (K_tl == bq.gp_log_S.Kxx).all()
    assert K_tl is bq.gp_log_S.Kxx

    K_del = bq.gp_Dc.Kxx
    bq._improve_covariance_conditioning(K_del)
    assert (K_del == bq.gp_Dc.Kxx).all()
    assert K_del is bq.gp_Dc.Kxx


def test_init():
    x, y = make_1d_gaussian()
    bq = BQ(x, y, gamma, ntry, n_candidate, s=0)
    assert (x == bq.R).all()
    assert (y == bq.S).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, gamma, ntry, n_candidate, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], gamma, ntry, n_candidate, s=0)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, gamma, ntry, n_candidate, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], gamma, ntry, n_candidate, s=0)


def test_log_transform():
    x, y = make_1d_gaussian()
    log_y = np.log((y / float(gamma)) + 1)
    bq = BQ(x, y, gamma, ntry, n_candidate, s=0)
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


def test_S_cov():
    raise NotImplementedError
