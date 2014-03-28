import matplotlib.pyplot as plt
import numpy as np
import pytest

from mental_rotation.model import BayesianQuadratureModel
from .test_base_model import TestBaseModel as BaseModel


class TestBayesianQuadratureModel(BaseModel):

    cls = BayesianQuadratureModel
    cls._iter = 3

    @pytest.mark.xfail(reason="not implemented")
    def test_log_S_i(self, model):
        raise NotImplementedError

    def test_S(self, model):
        model.sample()
        R = np.linspace(0, 2 * np.pi, 361)
        assert np.allclose(model.log_S(R, 0), np.log(model.S(R, 0)))
        assert np.allclose(model.log_S(R, 1), np.log(model.S(R, 1)))
        assert not np.allclose(model.S(R, 0), model.S(R, 1))

    @pytest.mark.xfail(reason="not implemented")
    def test_log_S(self, model):
        raise NotImplementedError

    def test_Z(self, model):
        model.sample()
        assert np.allclose(model.log_Z(0), np.log(model.Z(0)))
        assert np.allclose(model.log_Z(1), np.log(model.Z(1)))

    def test_log_lh(self, model):
        model.sample()
        assert (model.log_lh_h0 == model.log_Z(0)).all()
        assert (model.log_lh_h1 == model.log_Z(1)).all()

    @pytest.mark.xfail(reason="not implemented")
    def test_log_lh_h0(self, model):
        raise NotImplementedError

    @pytest.mark.xfail(reason="not implemented")
    def test_log_lh_h1(self, model):
        raise NotImplementedError

    def test_plot_bq(self, model):
        model.sample()
        model.plot_bq(0)
        plt.close('all')
        model.plot_bq(0, f_S=lambda R: np.ones_like(R))
        plt.close('all')

    def test_plot_all(self, model):
        model.sample()
        model.plot_all()
        plt.close('all')
        model.plot_all(f_S0=lambda R: np.ones_like(R), f_S1=lambda R: np.ones_like(R))
        plt.close('all')

    def test_S(self, model):
        model.sample()
        R = np.linspace(0, 2 * np.pi, 361)

        tl = model.log_S(R, 0)
        l = model.S(R, 0)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        tl = model.log_S(R, 1)
        l = model.S(R, 1)
        pos = l > 0
        assert np.allclose(tl[pos], np.log(l[pos]))

        assert not np.allclose(model.S(R, 0), model.S(R, 1))
