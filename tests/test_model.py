from mental_rotation.stimulus import Stimulus2D
from mental_rotation.model import model
from . import util

def test_prior():
    util.seed()
    for i in xrange(5):
        stim = Stimulus2D.random(3)
        assert model.log_prior(stim.vertices) == -2.9826069522587453
        stim = Stimulus2D.random(4)
        assert model.log_prior(stim.vertices) == -3.7218717299999811
        stim = Stimulus2D.random(5)
        assert model.log_prior(stim.vertices) == -4.173454435289436
        stim = Stimulus2D.random(6)
        assert model.log_prior(stim.vertices) == -4.40189358926468
        stim = Stimulus2D.random(7)
        assert model.log_prior(stim.vertices) == -4.4480111864459708
        stim = Stimulus2D.random(8)
        assert model.log_prior(stim.vertices) == -4.3399781038000036

