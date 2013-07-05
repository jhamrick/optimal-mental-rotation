from models import Model
import util
import numpy as np
from itertools import izip


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


def test_get_S():
    m = setup()
    for r, s in izip(m.R, m.S):
        d = np.round(np.degrees(r))
        s2 = m._get_S(d)
        assert np.allclose(s, s2), (s, s2)


def test_observed():
    m = setup()
    assert not m.observed(None)
    assert m.observed(0)
    assert m.observed(2*np.pi)
    assert not m.observed(np.pi)
    m.sample(np.pi)
    assert m.observed(np.pi)
    assert not m.observed(2.4590823476)
    m.sample(2.4590823476)
    assert m.observed(2.4590823476)


def test_sample():
    def check(m, r):
        s = m.sample(r)
        rw = r % (2*np.pi)
        d = int(np.round(np.degrees(r)))
        dw = int(np.round(np.degrees(rw)))
        assert (m._sampled == np.array([0, d])).all()
        assert (m.Ri == np.array([m.R[0], rw])).all()
        assert np.allclose(m.Si, np.array([m.S[0], s]))
        m.sample(r)
        assert (m._sampled == np.array([0, d, d])).all()
        assert (m.Ri == np.array([m.R[0], rw])).all()
        assert np.allclose(m.Si, np.array([m.S[0], s]))
        m.sample(rw)
        assert (m._sampled == np.array([0, d, d, dw])).all()
        assert (m.Ri == np.array([m.R[0], rw])).all()
        assert np.allclose(m.Si, np.array([m.S[0], s]))

    m0 = setup()
    for i in xrange(20):
        m = m0.copy()
        r = np.random.uniform(-2*np.pi, 2*np.pi)
        yield check, m, r


def test_curr_val():
    m = setup()
    for i in xrange(20):
        r0 = np.random.uniform(0, 2*np.pi)
        s0 = m.sample(r0)
        r, s = m.curr_val
        assert np.abs(r - r0) < np.radians(1)
        assert s == s0


def test_num_samples_left():
    m = setup()
    n = len(m._all_samples)
    print m.num_samples_left
    assert m.num_samples_left == (n - 1)
    for i, d in enumerate(m._all_samples.keys()):
        r = np.radians(d)
        m.sample(r)
        assert m.num_samples_left == (n - i - 1)


def test_next_val():
    m = setup()
    r0 = np.radians(1)
    m.sample(r0)
    r, s = m.next_val()
    assert r > r0

    r0 = np.radians(-1)
    m.sample(r0)
    r, s = m.next_val()
    assert r < r0
