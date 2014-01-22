import numpy as np
import pytest

from mental_rotation.model import OracleModel
from mental_rotation.model.model import log_similarity
from .test_base_model import TestBaseModel as BaseModel


class TestOracleModel(BaseModel):

    cls = OracleModel

    def test_log_S_i(self, Xa, Xb, model):
        model.sample()
        R = model.R_i
        F = model.F_i
        log_S = np.empty_like(R)

        for i, (r, f) in enumerate(zip(R, F)):
            Xr = Xa.copy_from_vertices()
            if f == 1:
                Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[i] = log_similarity(
                Xr.vertices, Xb.vertices, model.opts['S_sigma'])

        assert np.allclose(log_S, model.log_S_i)

    @pytest.mark.full
    def test_correct(self, theta, flipped, model):
        model.sample()

        assert np.isclose(model.target, model._unwrap(np.radians(theta)))
        assert model.model['F'].value == int(flipped)

        if theta == 0:
            assert model.direction == 0
        elif theta > 180:
            assert model.direction == -1
        else:
            assert model.direction == 1

    def test_R_i(self, model):
        model.sample()
        # at least once for F=0, at least once for F=1
        assert np.isclose(model.R_i, 0).sum() == 1

    def test_F_i(self, flipped, model):
        model.sample()
        assert (model.F_i == int(flipped)).all()
