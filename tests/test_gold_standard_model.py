import numpy as np
import pymc

from mental_rotation.model import GoldStandardModel
from .test_base_model import TestBaseModel
from . import util


class TestGoldStandardModel(TestBaseModel):

    cls = GoldStandardModel

    def test_R_i(self):
        super(TestGoldStandardModel, self).test_R_i()

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.empty(722)
        R[::2] = np.linspace(-np.pi, np.pi, 361)
        R[1::2] = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(R, m.R_i)

    def test_F_i(self):
        super(TestGoldStandardModel, self).test_R_i()

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        F = np.zeros(722)
        F[1::2] = 1
        assert np.allclose(F, m.F_i)

    def test_log_S_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        log_S = np.empty(R.size * 2)

        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S[2*i] = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

            Xr = Xa.copy_from_vertices()
            Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S[2*i + 1] = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])

        assert np.allclose(log_S, m.log_S_i)

    def test_log_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(m.log_S_i[::2], m.log_S(m.R_i[::2], 0))
        assert np.allclose(m.log_S_i[1::2], m.log_S(m.R_i[1::2], 1))
        assert np.allclose(m.log_S_i[::2], m.log_S(R, 0))
        assert np.allclose(m.log_S_i[1::2], m.log_S(R, 1))
        assert m.log_S(0, 0) == m.log_S(2 * np.pi, 0)
        assert m.log_S(0, 1) == m.log_S(2 * np.pi, 1)

    def test_log_dZ_dR_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        log_dZ_dR = np.empty(R.size * 2)

        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            log_S = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])
            log_p_R = pymc.distributions.von_mises_like(
                r, m.opts['R_mu'], m.opts['R_kappa'])
            log_p_F = pymc.distributions.bernoulli_like(0, 0.5)
            log_dZ_dR[2*i] = log_S + log_p_R + log_p_F

            Xr = Xa.copy_from_vertices()
            Xr.flip(np.array([0, 1]))
            Xr.rotate(np.degrees(r))
            log_S = m._log_similarity(
                Xr.vertices, Xb.vertices, m.opts['S_sigma'])
            log_p_R = pymc.distributions.von_mises_like(
                r, m.opts['R_mu'], m.opts['R_kappa'])
            log_p_F = pymc.distributions.bernoulli_like(1, 0.5)
            log_dZ_dR[2*i + 1] = log_S + log_p_R + log_p_F

        assert np.allclose(log_dZ_dR, m.log_dZ_dR_i)

    def test_log_dZ_dR(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(-np.pi, np.pi, 361)
        assert np.allclose(m.log_dZ_dR_i[::2], m.log_dZ_dR(m.R_i[::2], 0))
        assert np.allclose(m.log_dZ_dR_i[1::2], m.log_dZ_dR(m.R_i[1::2], 1))
        assert np.allclose(m.log_dZ_dR_i[::2], m.log_dZ_dR(R, 0))
        assert np.allclose(m.log_dZ_dR_i[1::2], m.log_dZ_dR(R, 1))

        log_p_R = np.array([
            pymc.distributions.von_mises_like(
                r, m.opts['R_mu'], m.opts['R_kappa'])
            for r in R])

        log_p_F = pymc.distributions.bernoulli_like(0, 0.5)
        log_dZ_dR = m.log_S(R, 0) + log_p_R + log_p_F
        assert np.allclose(log_dZ_dR, m.log_dZ_dR(R, 0))
        assert m.log_dZ_dR(0, 0) == m.log_dZ_dR(2 * np.pi, 0)

        log_p_F = pymc.distributions.bernoulli_like(0, 0.5)
        log_dZ_dR = m.log_S(R, 1) + log_p_R + log_p_F
        assert np.allclose(log_dZ_dR, m.log_dZ_dR(R, 1))
        assert m.log_dZ_dR(0, 1) == m.log_dZ_dR(2 * np.pi, 1)

    def test_log_Z(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(-np.pi, np.pi, 360)
        log_Z = np.log(np.trapz(m.dZ_dR(R, 0), R))
        assert np.allclose(log_Z, m.log_Z(0))
        log_Z = np.log(np.trapz(m.dZ_dR(R, 1), R))
        assert np.allclose(log_Z, m.log_Z(1))
