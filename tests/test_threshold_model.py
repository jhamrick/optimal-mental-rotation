import numpy as np
import pytest

from mental_rotation.model import ThresholdModel
from mental_rotation.model.model import log_similarity
from .test_base_model import TestBaseModel as BaseModel


class TestThresholdModel(BaseModel):

    cls = ThresholdModel

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

    @pytest.mark.smallrot
    def test_threshold(self, model):
        for i in xrange(200):
            model.status = "ready"
            model.sample()
            assert model.R_i.size < 10

    @pytest.mark.full
    def test_threshold_2(self, model):
        model.sample()
        assert model.R_i.size < 75
