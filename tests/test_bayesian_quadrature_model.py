import matplotlib.pyplot as plt
import numpy as np
import pytest

from mental_rotation.model import BayesianQuadratureModel
from .test_base_model import TestBaseModel as BaseModel
from . import util


class TestBayesianQuadratureModel(BaseModel):

    cls = BayesianQuadratureModel

    def test_log_S_i(self):
        pass

    def test_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(m.log_S(R, 0), np.log(m.S(R, 0)))
        assert np.allclose(m.log_S(R, 1), np.log(m.S(R, 1)))
        assert not np.allclose(m.S(R, 0), m.S(R, 1))

    def test_log_S(self):
        pass

    def test_Z(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_Z(0), np.log(m.Z(0)))
        assert np.allclose(m.log_Z(1), np.log(m.Z(1)))

    def test_log_lh_h0(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        log_p_Xa = m.model['Xa'].logp
        log_Z = m.log_Z(0)
        assert (m.log_lh_h0 == (log_p_Xa + log_Z)).all()
        log_Z = m.log_Z(1)
        assert (m.log_lh_h1 == (log_p_Xa + log_Z)).all()

    def test_log_lh_h1(self):
        pass

    def test_plot(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.plot()
        plt.close('all')

    def test_S(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)

        tl = m.log_S(R, 0)
        l = m.S(R, 0)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        tl = m.log_S(R, 1)
        l = m.S(R, 1)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        assert not np.allclose(m.S(R, 0), m.S(R, 1))


    def test_dZ_dR(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)

        tl = m.log_dZ_dR(R, 0)
        l = m.dZ_dR(R, 0)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        tl = m.log_dZ_dR(R, 1)
        l = m.dZ_dR(R, 1)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        assert not np.allclose(m.dZ_dR(R, 0), m.dZ_dR(R, 1))
