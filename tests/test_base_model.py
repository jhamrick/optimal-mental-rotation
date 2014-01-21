import numpy as np
import matplotlib.pyplot as plt
import pytest

from path import path
from copy import deepcopy

from mental_rotation.model import BaseModel
from mental_rotation.model.model import log_prior, log_similarity
from mental_rotation.stimulus import Stimulus2D
from . import util


class TestBaseModel(object):

    cls = BaseModel

    @pytest.fixture(scope="class")
    def config(self, request):
        tmp_path = path("/tmp/mental_rotation")
        config = util.setup_config(tmp_path)
        return config

    @pytest.fixture(scope="class", params=[(39, False), (39, True)])
    def stim(self, request, config):
        seed = config.getint("global", "seed")
        np.random.randint(seed)

        # create stimulus
        X = Stimulus2D.random(8)
        theta, flip = request.param
        if flip:
            X.flip([0, 1])
        X.rotate(theta)

        Xa = X.copy_from_initial()
        Xb = X.copy_from_vertices()
        return theta, flip, Xa, Xb

    @pytest.fixture
    def model(self, stim, config, request, tmpdir):
        # stimulus
        theta, flipped, Xa, Xb = stim

        # create the model
        m = self.cls(
            Xa.vertices, Xb.vertices, 
            S_sigma=config.getfloat('model', 'S_sigma'))

        # handle temporary directories
        self.pth = path(tmpdir.strpath).joinpath("save")
        def fin():
            self.pth.rmtree_p()
        request.addfinalizer(fin)

        return theta, flipped, Xa, Xb, m


    def test_priors(self, model):
        theta, flipped, Xa, Xb, m = model
        assert np.allclose(m.model['Xa'].value, Xa.vertices)
        assert np.allclose(m.model['Xb'].value, Xb.vertices)
        assert m.model['Xa'].logp == m.model['Xb'].logp
        assert m.model['Xa'].logp == log_prior(Xa.vertices)
        assert m.model['Xb'].logp == log_prior(Xb.vertices)

    def test_Xr(self, model):
        theta, flipped, Xa, Xb, m = model
        assert np.allclose(m.model['Xr'].value, Xa.vertices)
        assert np.allclose(m.model['Xr'].value, m.model['Xa'].value)

    def test_similarity(self, model):
        theta, flipped, Xa, Xb, m = model
        log_S = log_similarity(
            Xb.vertices,
            Xa.vertices,
            m.opts['S_sigma'])
        assert log_S == m.model['log_S'].logp

        log_S = log_similarity(
            m.model['Xb'].value,
            m.model['Xr'].value,
            m.opts['S_sigma'])
        assert log_S == m.model['log_S'].logp

    def test_plot(self, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.sample()

        fig, ax = plt.subplots()
        m.plot(ax, 0)
        m.plot(ax, 1)
        plt.close('all')

        fig, ax = plt.subplots()
        m.plot(ax, 0, f_S = lambda R, F: np.ones_like(R), color='g')
        plt.close('all')

    def test_R_i(self, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.sample()
        Ri = m.R_i
        # at least once for F=0, at least once for F=1
        assert np.isclose(Ri, 0).sum() >= 2

    def test_F_i(self, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.sample()
        Fi = m.F_i
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

        theta, flipped, Xa, Xb, m = model
        m.sample()
        assert np.allclose(m.log_S_i, np.log(m.S_i))

    def test_print_stats(self, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.sample()
        m.print_stats()

    def test_sample_and_save(self, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.sample()
        m.save(self.pth)

    def test_save(self, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.save(self.pth)

    def test_force_save(self, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.save(self.pth)
        with pytest.raises(IOError):
            m.save(self.pth)
        m.save(self.pth, force=True)

    def test_load(self, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        with pytest.raises(IOError):
            self.cls.load(self.pth)
        m.sample()
        m.save(self.pth)
        m2 = self.cls.load(self.pth)
        m2.print_stats()

    def test_load_and_sample(self, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
        m.save(self.pth)
        m2 = self.cls.load(self.pth)
        m2.sample()
        m2.print_stats()

    def test_sample_again(self, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb, m = model
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
