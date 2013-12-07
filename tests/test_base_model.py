import numpy as np
import matplotlib.pyplot as plt
from mental_rotation.model import BaseModel
from . import util


class TestBaseModel(object):

    cls = BaseModel

    def test_priors(self):
        Xa, Xb, m = util.make_model(self.cls)
        assert np.allclose(m.model['Xa'].value, Xa.vertices)
        assert np.allclose(m.model['Xb'].value, Xb.vertices)
        assert m.model['Xa'].logp == m.model['Xb'].logp
        assert m.model['Xa'].logp == m._prior(Xa.vertices)
        assert m.model['Xb'].logp == m._prior(Xb.vertices)

    def test_Xr(self):
        Xa, Xb, m = util.make_model(self.cls)
        assert np.allclose(m.model['Xr'].value, Xa.vertices)
        assert np.allclose(m.model['Xr'].value, m.model['Xa'].value)

    def test_similarity(self):
        Xa, Xb, m = util.make_model(self.cls)
        log_S = m._log_similarity(
            Xb.vertices,
            Xa.vertices,
            m.opts['S_sigma'])
        assert log_S == m.model['log_S'].logp

        log_S = m._log_similarity(
            m.model['Xb'].value,
            m.model['Xr'].value,
            m.opts['S_sigma'])
        assert log_S == m.model['log_S'].logp

    def test_plot(self):
        if self.cls is BaseModel:
            return

        fig, ax = plt.subplots()
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.plot(ax)
        plt.close('all')

    def test_R_i(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        Ri = m.R_i
        assert sorted(Ri)[0] == 0
        assert (Ri >= 0).all()
        assert (Ri < 2 * np.pi).all()

    def test_log_S_i(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_S_i(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_S_i, np.log(m.S_i))

    def test_log_S(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_S(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(m.log_S(R), np.log(m.S(R)))

    def test_log_dZ_dR_i(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_dZ_dR_i(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_dZ_dR_i, np.log(m.dZ_dR_i))

    def test_log_dZ_dR(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_dZ_dR(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(m.log_dZ_dR(R), np.log(m.dZ_dR(R)))

    def test_log_Z(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_Z(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_Z, np.log(m.Z))

    def test_log_lh_h0(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        log_p_Xa = m.model['Xa'].logp
        log_p_Xb = m.model['Xb'].logp
        assert m.log_lh_h0 == (log_p_Xa + log_p_Xb)

    def test_log_lh_h1(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        log_p_Xa = m.model['Xa'].logp
        log_Z = m.log_Z
        assert m.log_lh_h1 == (log_p_Xa + log_Z)

    def test_print_stats(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.print_stats()
