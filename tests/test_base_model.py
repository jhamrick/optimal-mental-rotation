import numpy as np
import matplotlib.pyplot as plt
import pytest

from path import path
from copy import deepcopy

from mental_rotation.model import BaseModel
from mental_rotation.model.model import log_prior, log_similarity


class TestBaseModel(object):

    cls = BaseModel

    def test_priors(self, Xa, Xb, model):
        assert np.allclose(model.model['Xa'].value, Xa.vertices)
        assert np.allclose(model.model['Xb'].value, Xb.vertices)
        assert model.model['Xa'].logp == model.model['Xb'].logp
        assert model.model['Xa'].logp == log_prior(Xa.vertices)
        assert model.model['Xb'].logp == log_prior(Xb.vertices)

    def test_Xr(self, Xa, Xb, model):
        assert np.allclose(model.model['Xr'].value, Xa.vertices)
        assert np.allclose(model.model['Xr'].value, model.model['Xa'].value)

    def test_similarity(self, Xa, Xb, model):
        log_S = log_similarity(
            Xb.vertices,
            Xa.vertices,
            model.opts['S_sigma'])
        assert log_S == model.model['log_S'].logp

        log_S = log_similarity(
            model.model['Xb'].value,
            model.model['Xr'].value,
            model.opts['S_sigma'])
        assert log_S == model.model['log_S'].logp

    def test_plot(self, model):
        if self.cls is BaseModel:
            return

        model.sample()

        fig, ax = plt.subplots()
        model.plot(ax, 0)
        model.plot(ax, 1)
        plt.close('all')

        fig, ax = plt.subplots()
        model.plot(ax, 0, f_S = lambda R, F: np.ones_like(R), color='g')
        plt.close('all')

    def test_R_i(self, model):
        if self.cls is BaseModel:
            return

        model.sample()
        Ri = model.R_i
        # at least once for F=0, at least once for F=1
        assert np.isclose(Ri, 0).sum() >= 2

    def test_F_i(self, model):
        if self.cls is BaseModel:
            return

        model.sample()
        Fi = model.F_i
        assert ((Fi == 0) | (Fi == 1)).all()
        assert (Fi == 0).any()
        assert (Fi == 1).any()

    def test_log_S_i(self, model):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_S_i(self, model):
        if self.cls is BaseModel:
            return

        model.sample()
        assert np.allclose(model.log_S_i, np.log(model.S_i))

    def test_print_stats(self, model):
        if self.cls is BaseModel:
            return

        model.sample()
        model.print_stats()

    def test_sample_and_save(self, model, tmppath):
        if self.cls is BaseModel:
            return

        model.sample()
        model.save(tmppath)

    def test_save(self, model, tmppath):
        if self.cls is BaseModel:
            return

        model.save(tmppath)

    def test_force_save(self, model, tmppath):
        if self.cls is BaseModel:
            return

        model.save(tmppath)
        with pytest.raises(IOError):
            model.save(tmppath)
        model.save(tmppath, force=True)

    def test_load(self, model, tmppath):
        if self.cls is BaseModel:
            return

        with pytest.raises(IOError):
            self.cls.load(tmppath)

        model.sample()
        model.save(tmppath)

        m2 = self.cls.load(tmppath)
        m2.print_stats()

    def test_load_and_sample(self, model, tmppath):
        if self.cls is BaseModel:
            return

        model.save(tmppath)
        m2 = self.cls.load(tmppath)
        m2.sample()
        m2.print_stats()

    def test_sample_again(self, model):
        if self.cls is BaseModel:
            return

        model.sample()
        state1 = deepcopy(model.__getstate__())
        model.sample()
        state2 = model.__getstate__()

        assert len(state1) == len(state2)
        for key in state1:
            if isinstance(state1[key], np.ndarray):
                assert (state1[key] == state2[key]).all()
            elif key == "_traces":
                for trace in state1[key]:
                    assert (state1[key][trace] == state2[key][trace]).all()
            else:
                assert state1[key] == state2[key]
