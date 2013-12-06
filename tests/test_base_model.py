import numpy as np
from mental_rotation.model import BaseModel
from . import util


def test_priors():
    Xa, Xb, m = util.make_model(BaseModel, 39, True)
    assert np.allclose(m.model['Xa'].value, Xa.vertices)
    assert np.allclose(m.model['Xb'].value, Xb.vertices)
    assert m.model['Xa'].logp == m.model['Xb'].logp
    assert m.model['Xa'].logp == m._prior(Xa.vertices)
    assert m.model['Xb'].logp == m._prior(Xb.vertices)
