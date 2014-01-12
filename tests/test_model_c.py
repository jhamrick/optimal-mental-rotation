import numpy as np
from mental_rotation.stimulus import Stimulus2D
from mental_rotation.model import model_c
from mental_rotation.model import model
from . import util

def test_log_prior():
    util.seed()
    for i in xrange(5):
        stim = Stimulus2D.random(3).vertices
        assert model_c.log_prior(stim) == -2.9826069522587453
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)

        stim = Stimulus2D.random(4).vertices
        assert model_c.log_prior(stim) == -3.7218717299999811
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)

        stim = Stimulus2D.random(5).vertices
        assert model_c.log_prior(stim) == -4.173454435289436
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)

        stim = Stimulus2D.random(6).vertices
        assert model_c.log_prior(stim) == -4.40189358926468
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)

        stim = Stimulus2D.random(7).vertices
        assert model_c.log_prior(stim) == -4.4480111864459708
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)

        stim = Stimulus2D.random(8).vertices
        assert model_c.log_prior(stim) == -4.3399781038000036
        assert model_c.log_prior(stim) == model.slow_log_prior(stim)


def test_log_factorial():
    assert np.allclose(model_c.log_factorial(3), 1.791759469228055)
    assert np.allclose(model_c.log_factorial(4), 3.1780538303479458)
    assert np.allclose(model_c.log_factorial(5), 4.7874917427820458)


def test_log_similarity():
    util.seed()
    for i in xrange(5):
        stim1 = Stimulus2D.random(6).vertices
        stim2 = Stimulus2D.random(6).vertices
        s1 = model_c.log_similarity(stim1, stim2, 0.15)
        s2 = model.slow_log_similarity(stim1, stim2, 0.15)
        assert np.allclose(s1, s2)

