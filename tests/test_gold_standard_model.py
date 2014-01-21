import numpy as np
import pymc

from mental_rotation.model import GoldStandardModel
from mental_rotation.model.model import log_similarity
from .test_base_model import TestBaseModel as BaseModel
from . import util


class TestGoldStandardModel(BaseModel):

    cls = GoldStandardModel

    def test_R_i(self, basic_stim, model):
        super(TestGoldStandardModel, self).test_R_i(basic_stim, model)

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        R = np.empty(722)
        R[::2] = np.linspace(-np.pi, np.pi, 361)
        R[1::2] = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(R, m.R_i)

    def test_F_i(self, basic_stim, model):
        super(TestGoldStandardModel, self).test_R_i(basic_stim, model)

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        F = np.zeros(722)
        F[1::2] = 1
        assert np.allclose(F, m.F_i)

    def test_log_S_i(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        log_S = np.empty(R.size * 2)

        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S[2*i] = log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

            Xr = Xa.copy_from_vertices()
            Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[2*i + 1] = log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

        assert np.allclose(log_S, m.log_S_i)

    def test_S(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(m.log_S(R, 0), np.log(m.S(R, 0)))
        assert np.allclose(m.log_S(R, 1), np.log(m.S(R, 1)))
        assert not np.allclose(m.S(R, 0), m.S(R, 1))

    def test_log_S(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(m.log_S_i[::2], m.log_S(m.R_i[::2], 0))
        assert np.allclose(m.log_S_i[1::2], m.log_S(m.R_i[1::2], 1))
        assert np.allclose(m.log_S_i[::2], m.log_S(R, 0))
        assert np.allclose(m.log_S_i[1::2], m.log_S(R, 1))
        assert m.log_S(0, 0) == m.log_S(2 * np.pi, 0)
        assert m.log_S(0, 1) == m.log_S(2 * np.pi, 1)

    def test_hypothesis(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        if flipped:
            assert m.hypothesis_test() == 1
        else:
            assert m.hypothesis_test() == 0
