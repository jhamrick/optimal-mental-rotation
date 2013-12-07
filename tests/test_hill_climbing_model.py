import numpy as np
import pymc

from mental_rotation.model import HillClimbingModel
from .test_base_model import TestBaseModel
from . import util


class TestHillClimbingModel(TestBaseModel):

    cls = HillClimbingModel

    def test_S_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        S = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            S[i] = np.exp(m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma))
        assert np.allclose(S, m.S_i)

    def test_p_i(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = m.R_i
        p = np.empty_like(R)
        Xr = Xa.copy_from_vertices()
        logp_Xa = m._prior(Xa.vertices)
        logp_Xb = m._prior(Xb.vertices)
        for i, r in enumerate(R):
            Xr = Xa.copy_from_vertices()
            Xr.rotate(np.degrees(r))
            logS = m._log_similarity(
                Xr.vertices, Xb.vertices, util.S_sigma)
            logp_R = pymc.distributions.von_mises_like(
                r, util.R_mu, util.R_kappa)
            p[i] = np.exp(logS + logp_R + logp_Xa + logp_Xb)
        assert np.allclose(p, m.p_i)

    def test_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.S_i, m.S(m.R_i))

    def test_p(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.p_i, m.p(m.R_i))

    def test_integrate(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.radians(np.arange(0, 360))
        Z = np.trapz(m.p(R), R)
        assert np.allclose(Z, m.integrate())
