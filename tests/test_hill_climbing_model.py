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
        F = m.F_i
        log_S = np.empty_like(R)

        for i, (r, f) in enumerate(zip(R, F)):
            Xr = Xa.copy_from_vertices()
            if f == 1:
                Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[i] = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

        assert np.allclose(log_S, m.log_S_i)

    def test_log_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        F = m.F_i
        log_S = np.empty_like(R)
        log_S[F == 0] = m.log_S(m.R_i[F == 0], 0)
        log_S[F == 1] = m.log_S(m.R_i[F == 1], 1)
        assert np.allclose(m.log_S_i, log_S)
        assert m.log_S(0, 0) == m.log_S(2 * np.pi, 0)
        assert m.log_S(0, 1) == m.log_S(2 * np.pi, 1)

    def test_log_dZ_dR_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        F = m.F_i
        log_dZ_dR = np.empty_like(R)

        for i, (r, f) in enumerate(zip(R, F)):
            Xr = Xa.copy_from_vertices()
            if f == 1:
                Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])
            log_p_R = pymc.distributions.von_mises_like(
                r, m.opts['R_mu'], m.opts['R_kappa'])
            log_p_F = pymc.distributions.bernoulli_like(f, 0.5)
            log_dZ_dR[i] = log_S + log_p_R + log_p_F

        assert np.allclose(log_dZ_dR, m.log_dZ_dR_i)

    def test_log_dZ_dR(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        F = m.F_i
        log_dZ_dR = np.empty_like(R)
        log_dZ_dR[F == 0] = m.log_dZ_dR(R[F == 0], 0)
        log_dZ_dR[F == 1] = m.log_dZ_dR(R[F == 1], 1)
        assert np.allclose(m.log_dZ_dR_i, log_dZ_dR)
        assert m.log_dZ_dR(0, 0) == m.log_dZ_dR(2 * np.pi, 0)
        assert m.log_dZ_dR(0, 1) == m.log_dZ_dR(2 * np.pi, 1)

    def test_log_Z(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 360)
        log_Z0 = np.log(np.trapz(m.dZ_dR(R, 0), R))
        log_Z1 = np.log(np.trapz(m.dZ_dR(R, 1), R))
        assert np.allclose(log_Z0, m.log_Z(0))
        assert np.allclose(log_Z1, m.log_Z(1))
