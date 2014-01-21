import numpy as np

from mental_rotation.model import OracleModel, model
from .test_base_model import TestBaseModel as BaseModel
from . import util


class TestOracleModel(BaseModel):

    cls = OracleModel

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

    def test_correct(self):
        for theta in np.arange(0, 360, 20):
            for F in [False, True]:
                Xa, Xb, m = util.make_model(self.cls, theta=theta, flip=F)
                m.sample()

                assert np.isclose(m.target, m._unwrap(np.radians(theta)))
                assert m.model['F'].value == int(F)

                if theta == 0:
                    assert m.direction == 0
                elif theta > 180:
                    assert m.direction == -1
                else:
                    assert m.direction == 1

    def test_R_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        Ri = m.R_i
        # at least once for F=0, at least once for F=1
        assert np.isclose(Ri, 0).sum() == 1

    def test_F_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        Fi = m.F_i
        assert ((Fi == 0) | (Fi == 1)).all()
