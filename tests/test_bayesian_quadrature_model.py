import matplotlib.pyplot as plt
import numpy as np
import pytest

from mental_rotation.model import BayesianQuadratureModel
from .test_base_model import TestBaseModel
from . import util


class TestBayesianQuadratureModel(TestBaseModel):

    cls = BayesianQuadratureModel

    @pytest.mark.xfail
    def test_log_S_i(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="S is sometimes negative")
    def test_S(self):
        super(TestBayesianQuadratureModel, self).test_S()

    def test_log_S(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        assert np.allclose(
            np.exp(m.log_S_i), np.exp(m.log_S(m.R_i)), atol=1e-7)
        assert np.allclose(m.log_S(0), m.log_S(2 * np.pi))

    def test_log_dZ_dR_i(self):
        pass

    def test_log_dZ_dR(self):
        pass

    def test_dZ_dR(self):
        pass

    @pytest.mark.xfail
    def test_log_Z(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Z is sometimes negative")
    def test_Z(self):
        super(TestBayesianQuadratureModel, self).test_Z()

    @pytest.mark.xfail(reason="Z is sometimes negative")
    def test_log_lh_h1(self):
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        log_p_Xa = m.model['Xa'].logp
        log_Z = m.log_Z
        assert (m.log_lh_h1 == (log_p_Xa + log_Z)).all()

    def test_plot(self):
        Xa, Xb, m = util.make_model(BayesianQuadratureModel)
        m.sample()
        fig, axes = plt.subplots(2, 2)
        m.plot(axes)
        plt.close('all')
