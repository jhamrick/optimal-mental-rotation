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
        logS = m._log_similarity(
            Xb.vertices,
            Xa.vertices,
            util.S_sigma)
        assert logS == m.model['logS'].logp

        logS = m._log_similarity(
            m.model['Xb'].value,
            m.model['Xr'].value,
            util.S_sigma)
        assert logS == m.model['logS'].logp

    def test_plot(self):
        if self.cls is BaseModel:
            return

        fig, ax = plt.subplots()
        Xa, Xb, m = util.make_model(self.cls)
        m.sample()
        m.plot(ax)
        plt.show()
        plt.close('all')
