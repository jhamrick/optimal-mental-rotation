import numpy as np
import matplotlib.pyplot as plt
from path import path
from mental_rotation.model import BaseModel, model
from . import util


class TestBaseModel(object):

    cls = BaseModel

    def test_priors(self):
        Xa, Xb, m = util.make_model(self.cls)
        assert np.allclose(m.model['Xa'].value, Xa.vertices)
        assert np.allclose(m.model['Xb'].value, Xb.vertices)
        assert m.model['Xa'].logp == m.model['Xb'].logp
        assert m.model['Xa'].logp == model.prior(Xa.vertices)
        assert m.model['Xb'].logp == model.prior(Xb.vertices)

    def test_Xr(self):
        Xa, Xb, m = util.make_model(self.cls)
        assert np.allclose(m.model['Xr'].value, Xa.vertices)
        assert np.allclose(m.model['Xr'].value, m.model['Xa'].value)

    def test_similarity(self):
        Xa, Xb, m = util.make_model(self.cls)
        log_S = model.log_similarity(
            Xb.vertices,
            Xa.vertices,
            m.opts['S_sigma'])
        assert log_S == m.model['log_S'].logp

        log_S = model.log_similarity(
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
        # once for F=0, once for F=1
        assert np.isclose(Ri, 0).sum() == 2

    def test_F_i(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        Fi = m.F_i
        assert ((Fi == 0) | (Fi == 1)).all()
        assert (Fi == 0).any()
        assert (Fi == 1).any()

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
        assert np.allclose(m.log_S(R, 0), np.log(m.S(R, 0)))
        assert np.allclose(m.log_S(R, 1), np.log(m.S(R, 1)))
        assert not np.allclose(m.S(R, 0), m.S(R, 1))

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
        assert np.allclose(m.log_dZ_dR(R, 0), np.log(m.dZ_dR(R, 0)))
        assert np.allclose(m.log_dZ_dR(R, 1), np.log(m.dZ_dR(R, 1)))
        assert not np.allclose(m.dZ_dR(R, 0), m.dZ_dR(R, 1))

    def test_log_Z(self):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_Z(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(m.log_Z(0), np.log(m.Z(0)))
        assert np.allclose(m.log_Z(1), np.log(m.Z(1)))

    def test_log_lh_h0(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        log_p_Xa = m.model['Xa'].logp
        log_Z = m.log_Z(0)
        assert m.log_lh_h0 == (log_p_Xa + log_Z)

    def test_log_lh_h1(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        log_p_Xa = m.model['Xa'].logp
        log_Z = m.log_Z(1)
        assert m.log_lh_h1 == (log_p_Xa + log_Z)

    def test_print_stats(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.print_stats()

    # def test_sample_and_close(self):
    #     if self.cls is BaseModel:
    #         return

    #     try:
    #         Xa, Xb, m = util.make_model(self.cls, name='test')
    #         m.sample()
    #         m.close()
    #     except:
    #         raise
    #     finally:
    #         pth = path("test.hdf5")
    #         print pth.abspath()
    #         if pth.exists():
    #             pth.remove()

    # def test_close(self):
    #     if self.cls is BaseModel:
    #         return

    #     try:
    #         Xa, Xb, m = util.make_model(self.cls, name='test')
    #         m.close()
    #     except:
    #         raise
    #     finally:
    #         pth = path("test.hdf5")
    #         print pth.abspath()
    #         if pth.exists():
    #             pth.remove()

    # def test_load(self):
    #     if self.cls is BaseModel:
    #         return

    #     try:
    #         Xa, Xb, m = util.make_model(self.cls, name='test')
    #         m.sample()
    #         m.close()
    #         m2 = self.cls.load('test')
    #         m2.print_stats()
    #         m2.close()
    #     except:
    #         raise
    #     finally:
    #         pth = path("test.hdf5")
    #         if pth.exists():
    #             pth.remove()

    # def test_load_and_sample(self):
    #     if self.cls is BaseModel:
    #         return

    #     try:
    #         Xa, Xb, m = util.make_model(self.cls, name='test')
    #         m.close()
    #         m2 = self.cls.load('test')
    #         m2.sample()
    #         m2.print_stats()
    #         m2.close()
    #     except:
    #         raise
    #     finally:
    #         pth = path("test.hdf5")
    #         if pth.exists():
    #             pth.remove()
