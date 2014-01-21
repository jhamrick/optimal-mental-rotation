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

    basic_stim = pytest.fixture(scope="class", params=[(39, False), (39, True)])(stim)
    full_stim = pytest.fixture(scope="class", params=[(t, f) for t in range(0, 360, 45) for f in [True, False]])(stim)

    @pytest.fixture
    def model(self, config, request, tmpdir):
        S_sigma = config.getfloat('model', 'S_sigma')

        # create the model
        def make_model(Xa, Xb):
            m = self.cls(
                Xa.vertices, Xb.vertices, 
                S_sigma=S_sigma)
            return m

        # handle temporary directories
        self.pth = path(tmpdir.strpath).joinpath("save")
        def fin():
            self.pth.rmtree_p()
        request.addfinalizer(fin)

        return make_model

    def test_priors(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        assert np.allclose(m.model['Xa'].value, Xa.vertices)
        assert np.allclose(m.model['Xb'].value, Xb.vertices)
        assert m.model['Xa'].logp == m.model['Xb'].logp
        assert m.model['Xa'].logp == log_prior(Xa.vertices)
        assert m.model['Xb'].logp == log_prior(Xb.vertices)

    def test_Xr(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        assert np.allclose(m.model['Xr'].value, Xa.vertices)
        assert np.allclose(m.model['Xr'].value, m.model['Xa'].value)

    def test_similarity(self, basic_stim, model):
        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

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

    def test_plot(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()

        fig, ax = plt.subplots()
        m.plot(ax, 0)
        m.plot(ax, 1)
        plt.close('all')

        fig, ax = plt.subplots()
        m.plot(ax, 0, f_S = lambda R, F: np.ones_like(R), color='g')
        plt.close('all')

    def test_R_i(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        Ri = m.R_i
        # at least once for F=0, at least once for F=1
        assert np.isclose(Ri, 0).sum() >= 2

    def test_F_i(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        Fi = m.F_i
        assert ((Fi == 0) | (Fi == 1)).all()
        assert (Fi == 0).any()
        assert (Fi == 1).any()

    def test_log_S_i(self, basic_stim, model):
        if self.cls is BaseModel:
            return
        raise NotImplementedError

    def test_S_i(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        assert np.allclose(m.log_S_i, np.log(m.S_i))

    def test_print_stats(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        m.print_stats()

    def test_sample_and_save(self, basic_stim, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.sample()
        m.save(self.pth)

    def test_save(self, basic_stim, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.save(self.pth)

    def test_force_save(self, basic_stim, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.save(self.pth)
        with pytest.raises(IOError):
            m.save(self.pth)
        m.save(self.pth, force=True)

    def test_load(self, basic_stim, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        with pytest.raises(IOError):
            self.cls.load(self.pth)
        m.sample()
        m.save(self.pth)
        m2 = self.cls.load(self.pth)
        m2.print_stats()

    def test_load_and_sample(self, basic_stim, model, tmpdir):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

        m.save(self.pth)
        m2 = self.cls.load(self.pth)
        m2.sample()
        m2.print_stats()

    def test_sample_again(self, basic_stim, model):
        if self.cls is BaseModel:
            return

        theta, flipped, Xa, Xb = basic_stim
        m = model(Xa, Xb)

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
