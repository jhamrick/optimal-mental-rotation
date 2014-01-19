import numpy as np

from mental_rotation.model import ThresholdModel, model
from .test_base_model import TestBaseModel as BaseModel
from . import util


class TestThresholdModel(BaseModel):

    cls = ThresholdModel

    def test_log_S_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        F = m.F_i
        log_S = np.empty_like(R)

        for i, (r, f) in enumerate(zip(R, F)):
            Xr = Xa.copy_from_vertices()
            if f == 1:
                Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[i] = model.log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

        assert np.allclose(log_S, m.log_S_i)
