from models import Model
import util
import numpy as np


def load():
    opt = util.load_opt()
    hyp = opt['example_hyp']
    num = opt['example_num']
    rot = opt['example_rot']
    stimname = '%s_%s_h%d' % (num, rot, hyp)
    out = util.load_stimulus(stimname, opt['stim_dir'])
    return opt, out


def setup():
    opt, out = load()
    theta, Xa, Xb, Xm, Ia, Ib, Im, R = out
    m = Model(Ia[0], Ib[0], Im[:, 0], R, **opt)
    return m


def test_init():
    opt, out = load()
    theta, Xa, Xb, Xm, Ia, Ib, Im, R = out
    m = Model(Ia[0], Ib[0], Im[:, 0], R, **opt)
    assert m.opt is not opt
    assert (m.Xa == Xa).all()
    assert (m.Xb == Xb).all()
    assert (m.Xm == Xm).all()
    assert (m.R == R).all()
    assert m.S.shape == m.R.shape
    assert (m._sampled == 0).all()
    assert (m.Ri == R[0]).all()
    assert np.allclose(m.Si, m.S[0])
    assert m.S_mean is None
    assert m.S_var is None
    assert m.Z_mean is None
    assert m.Z_var is None
