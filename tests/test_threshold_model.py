import numpy as np

from mental_rotation.model import ThresholdModel
from mental_rotation.model.model import log_similarity
from .test_base_model import TestBaseModel as BaseModel
from . import util


class TestThresholdModel(BaseModel):

    cls = ThresholdModel

    def test_log_S_i(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)
        m.sample()

        R = m.R_i
        F = m.F_i
        log_S = np.empty_like(R)

        for i, (r, f) in enumerate(zip(R, F)):
            Xr = Xa.copy_from_vertices()
            if f == 1:
                Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[i] = log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

        assert np.allclose(log_S, m.log_S_i)
