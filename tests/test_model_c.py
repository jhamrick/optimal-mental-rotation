import numpy as np
from mental_rotation.stimulus import Stimulus2D
from mental_rotation.model import model_c
from . import util

def test_log_prior():
    util.seed()
    for i in xrange(5):
        stim = Stimulus2D.random(3)
        assert model_c.log_prior(stim.vertices) == -2.9826069522587453
        stim = Stimulus2D.random(4)
        assert model_c.log_prior(stim.vertices) == -3.7218717299999811
        stim = Stimulus2D.random(5)
        assert model_c.log_prior(stim.vertices) == -4.173454435289436
        stim = Stimulus2D.random(6)
        assert model_c.log_prior(stim.vertices) == -4.40189358926468
        stim = Stimulus2D.random(7)
        assert model_c.log_prior(stim.vertices) == -4.4480111864459708
        stim = Stimulus2D.random(8)
        assert model_c.log_prior(stim.vertices) == -4.3399781038000036


def test_log_factorial():
    assert np.allclose(model_c.log_factorial(3), 1.791759469228055)
    assert np.allclose(model_c.log_factorial(4), 3.1780538303479458)
    assert np.allclose(model_c.log_factorial(5), 4.7874917427820458)
