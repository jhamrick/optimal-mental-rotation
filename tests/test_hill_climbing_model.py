import numpy as np
import pymc

from mental_rotation.model import HillClimbingModel
from .test_base_model import TestBaseModel
from . import util


class TestHillClimbingModel(TestBaseModel):

    cls = HillClimbingModel

    def test_log_S_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        log_S = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S[i] = m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma)
        assert np.allclose(log_S, m.log_S_i)

    def test_log_S_i_circle(self):
        Xa = util.make_circle()
        Xb = Xa.copy_from_vertices()
        m = HillClimbingModel(Xa, Xb, util.R_mu, util.R_kappa, util.S_sigma)
        m.sample()
        assert np.allclose(m.log_S_i, m.log_S_i[0])

        m = HillClimbingModel(Xa, Xb, util.R_mu, util.R_kappa, util.S_sigma)
        m.direction = 1
        m.sample()
        assert np.allclose(m.log_S_i, m.log_S_i[0])

        m = HillClimbingModel(Xa, Xb, util.R_mu, util.R_kappa, util.S_sigma)
        m.direction = -1
        m.sample()
        assert np.allclose(m.log_S_i, m.log_S_i[0])

    def test_log_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_S_i, m.log_S(m.R_i))
        assert m.log_S(0) == m.log_S(2 * np.pi)

    def test_log_dZ_dR_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        log_dZ_dR = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S = m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma)
            log_p_R = pymc.distributions.von_mises_like(
                r, util.R_mu, util.R_kappa)
            log_dZ_dR[i] = log_S + log_p_R
        assert np.allclose(log_dZ_dR, m.log_dZ_dR_i)

    def test_log_dZ_dR(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_dZ_dR_i, m.log_dZ_dR(m.R_i))
        assert m.log_dZ_dR(0) == m.log_dZ_dR(2 * np.pi)

    def test_log_Z(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 360)
        log_Z = np.log(np.trapz(m.dZ_dR(R), R))
        assert np.allclose(log_Z, m.log_Z)
