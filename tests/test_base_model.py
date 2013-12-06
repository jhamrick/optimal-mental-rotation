import numpy as np
from mental_rotation.model import BaseModel
from . import util


def test_priors():
    # for R in np.linspace(0, 360, 100):
    #     for flip in [True, False]:
    Xa, Xb, m = util.make_model(BaseModel, 90, False)
    assert np.allclose(m.model['Xa'].value, Xa.vertices)
    assert np.allclose(m.model['Xb'].value, Xb.vertices)
    assert m.model['Xa'].logp == m.model['Xb'].logp
