import numpy as np
import pymc

from mental_rotation.model import GoldStandardModel
from mental_rotation.model.model import log_similarity
from .test_base_model import TestBaseModel as BaseModel


class TestGoldStandardModel(BaseModel):

    cls = GoldStandardModel

    def test_R_i_2(self, model):
        model.sample()
        R = np.empty(722)
        R[::2] = np.linspace(-np.pi, np.pi, 361)
        R[1::2] = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(R, model.R_i)

    def test_F_i_2(self, model):
        model.sample()
        F = np.zeros(722)
        F[1::2] = 1
        assert np.allclose(F, model.F_i)

    def test_log_S_i(self, Xa, Xb, model):
        model.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        log_S = np.empty(R.size * 2)

        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S[2*i] = log_similarity(
                Xr.vertices, Xb.vertices, model.opts['S_sigma'])

            Xr = Xa.copy_from_vertices()
            Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[2*i + 1] = log_similarity(
                Xr.vertices, Xb.vertices, model.opts['S_sigma'])

        assert np.allclose(log_S, model.log_S_i)

    def test_S(self, model):
        model.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(model.log_S(R, 0), np.log(model.S(R, 0)))
        assert np.allclose(model.log_S(R, 1), np.log(model.S(R, 1)))
        assert not np.allclose(model.S(R, 0), model.S(R, 1))

    def test_log_S(self, model):
        model.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(model.log_S_i[::2], model.log_S(model.R_i[::2], 0))
        assert np.allclose(model.log_S_i[1::2], model.log_S(model.R_i[1::2], 1))
        assert np.allclose(model.log_S_i[::2], model.log_S(R, 0))
        assert np.allclose(model.log_S_i[1::2], model.log_S(R, 1))
        assert model.log_S(0, 0) == model.log_S(2 * np.pi, 0)
        assert model.log_S(0, 1) == model.log_S(2 * np.pi, 1)

    def test_hypothesis(self, flipped, model):
        model.sample()
        assert model.hypothesis_test() == int(flipped)
