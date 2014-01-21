import numpy as np
import matplotlib.pyplot as plt
from path import path
from mental_rotation.model import BaseModel, model
from . import util
from copy import deepcopy
import pytest


class TestBaseModel(object):

    cls = BaseModel

    def test_priors(self):
        Xa, Xb, m = util.make_model(self.cls)
        assert np.allclose(m.model['Xa'].value, Xa.vertices)
        assert np.allclose(m.model['Xb'].value, Xb.vertices)
        assert m.model['Xa'].logp == m.model['Xb'].logp
        assert m.model['Xa'].logp == model.log_prior(Xa.vertices)
        assert m.model['Xb'].logp == model.log_prior(Xb.vertices)

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

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()

        fig, ax = plt.subplots()
        m.plot(ax, 0)
        m.plot(ax, 1)
        plt.close('all')

        fig, ax = plt.subplots()
        m.plot(ax, 0, f_S = lambda R, F: np.ones_like(R), color='g')
        plt.close('all')

    def test_R_i(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        Ri = m.R_i
        # at least once for F=0, at least once for F=1
        assert np.isclose(Ri, 0).sum() >= 2

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

    def test_print_stats(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.print_stats()

    def test_sample_and_save(self):
        if self.cls is BaseModel:
            return

        pth = path("/tmp/test_model")

        try:
            Xa, Xb, m = util.make_model(self.cls)
            m.sample()
            m.save(pth)
        except:
            raise
        finally:
            if pth.exists():
                pth.rmtree_p()

    def test_save(self):
        if self.cls is BaseModel:
            return

        pth = path("/tmp/test_model")
        Xa, Xb, m = util.make_model(self.cls, name='test')

        try:
            m.save(pth)
        except:
            raise
        finally:
            if pth.exists():
                pth.rmtree_p()

    def test_force_save(self):
        if self.cls is BaseModel:
            return

        pth = path("/tmp/test_model")
        if not pth.exists():
            pth.mkdir_p()

        Xa, Xb, m = util.make_model(self.cls, name='test')
        with pytest.raises(IOError):
            m.save(pth)

        try:
            m.save(pth, force=True)
        except:
            raise
        finally:
            if pth.exists():
                pth.rmtree_p()

    def test_load(self):
        if self.cls is BaseModel:
            return

        pth = path("/tmp/test_model")
        Xa, Xb, m = util.make_model(self.cls)

        with pytest.raises(IOError):
            self.cls.load(pth)
            
        try:
            m.sample()
            m.save(pth)
            m2 = self.cls.load(pth)
            m2.print_stats()
        except:
            raise
        finally:
            if pth.exists():
                pth.rmtree_p()

    def test_load_and_sample(self):
        if self.cls is BaseModel:
            return

        pth = path("/tmp/test_model")
        Xa, Xb, m = util.make_model(self.cls)
            
        try:
            m.save(pth)
            m2 = self.cls.load(pth)
            m2.sample()
            m2.print_stats()
        except:
            raise
        finally:
            if pth.exists():
                pth.rmtree_p()

    def test_sample_again(self):
        if self.cls is BaseModel:
            return

        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        state1 = deepcopy(m.__getstate__())
        m.sample()
        state2 = m.__getstate__()

        assert len(state1) == len(state2)
        for key in state1:
            if isinstance(state1[key], np.ndarray):
                assert (state1[key] == state2[key]).all()
            elif key == "_traces":
                for trace in state1[key]:
                    assert (state1[key][trace] == state2[key][trace]).all()
            else:
                assert state1[key] == state2[key]
