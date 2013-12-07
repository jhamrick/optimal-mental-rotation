import numpy as np
import pymc

from mental_rotation.model import GoldStandardModel
from .test_base_model import TestBaseModel
from . import util


class TestGoldStandardModel(TestBaseModel):

    cls = GoldStandardModel

    def test_R_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        assert np.allclose(R, m.R_i)

    def test_S_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        S = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        for i, r in enumerate(R):
            Xr = Xr.copy_from_vertices()
            S[i] = np.exp(m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma))
            Xr.rotate(1)
        assert np.allclose(S, m.S_i)

    def test_p_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        p = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        logp_Xa = m._prior(Xa.vertices)
        logp_Xb = m._prior(Xb.vertices)
        for i, r in enumerate(R):
            Xr = Xr.copy_from_vertices()
            logS = m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma)
            logp_R = pymc.distributions.von_mises_like(
                r, util.R_mu, util.R_kappa)
            p[i] = np.exp(logS + logp_R + logp_Xa + logp_Xb)
            Xr.rotate(1)
        assert np.allclose(p, m.p_i)

    def test_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        assert np.allclose(m.S_i, m.S(m.R_i))
        assert np.allclose(m.S_i, m.S(R))

    def test_p(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        assert np.allclose(m.p_i, m.p(m.R_i))
        assert np.allclose(m.p_i, m.p(R))

        logp_Xa = m._prior(Xa.vertices)
        logp_Xb = m._prior(Xb.vertices)
        logp_R = np.array([
            pymc.distributions.von_mises_like(r, util.R_mu, util.R_kappa)
            for r in R])

        p = np.exp(np.log(m.S_i) + logp_R + logp_Xa + logp_Xb)
        assert np.allclose(p, m.p(R))

    def test_integrate(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        Z = np.trapz(m.p(R), R)
        assert np.allclose(Z, m.integrate())
