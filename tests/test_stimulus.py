import numpy as np
import matplotlib.pyplot as plt
import pytest
from tempfile import NamedTemporaryFile
from mental_rotation import Stimulus2D

from . import util


def test_vertices_ndim():
    util.seed()
    vertices = np.random.rand(8)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_shape():
    util.seed()
    vertices = np.random.rand(8, 3)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_copy():
    util.seed()
    vertices = np.random.rand(8, 2)
    stim = Stimulus2D(vertices)
    assert vertices is not stim._v


def test_random():
    stim = Stimulus2D.random(8)
    assert stim._v.shape[0] == 8

    for i in xrange(20):
        stim = Stimulus2D.random((6, 8))
        n = stim._v.shape[0]
        assert n >= 6 and n <= 8

    with pytest.raises(ValueError):
        stim = Stimulus2D.random((8, 6))

    with pytest.raises(ValueError):
        stim = Stimulus2D.random((8, 6, 8))

    with pytest.raises(ValueError):
        stim = Stimulus2D.random((8,))


def test_rotate():
    stim = util.make_stim()
    stim.rotate(90)
    assert stim.operations == [["rotate", 90.0]]
    assert (stim.vertices != stim._v).any()


def test_rotate4():
    stim = util.make_stim()
    stim.rotate(90)
    stim.rotate(90)
    stim.rotate(90)
    stim.rotate(90)
    assert stim.operations == [["rotate", 90.0]]*4
    assert np.allclose(stim.vertices, stim._v)


def test_flip2():
    stim = util.make_stim()
    stim.flip([1, 0])
    stim.flip([1, 0])
    assert stim.operations == [["flip", [1.0, 0.0]]]*2
    assert np.allclose(stim.vertices, stim._v)


def test_equality():
    stim1 = util.make_stim()
    stim2 = util.make_stim()
    assert stim1 == stim1
    assert stim2 == stim2
    assert stim1 == stim2
    assert not (stim1 != stim2)

    stim2.rotate(90)
    assert stim1 != stim2
    v = stim2.vertices
    stim2.rotate(90)
    stim2.rotate(90)
    stim2.rotate(90)
    assert stim1 != stim2

    stim2 = util.make_stim()
    assert stim1 == stim2
    stim2._v[:] = v
    assert stim1 != stim2


def test_save():
    fh = NamedTemporaryFile()
    stim = util.make_stim()
    with pytest.raises(IOError):
        stim.save(fh.name, force=False)
    stim.save(fh.name, force=True)


def test_io():
    fh = NamedTemporaryFile()
    stim1 = util.make_stim()
    stim1.save(fh.name, force=True)
    stim2 = Stimulus2D.load(fh.name)
    assert stim1 == stim2


def test_copy_from_state():
    stim1 = util.make_stim()
    stim1.rotate(90)
    stim2 = stim1.copy_from_state()
    assert stim1 == stim2
    assert stim1 is not stim2
    stim1.rotate(90)
    assert stim1 != stim2


def test_copy_from_vertices():
    stim1 = util.make_stim()
    stim1.rotate(90)
    stim2 = stim1.copy_from_vertices()
    assert stim1 != stim2
    assert np.allclose(stim1.vertices, stim2.vertices)
    assert np.allclose(stim1.vertices, stim2._v)
    assert not np.allclose(stim1._v, stim2.vertices)
    assert not np.allclose(stim1._v, stim2._v)


def test_copy_from_initial():
    stim1 = util.make_stim()
    stim1.rotate(90)
    stim2 = stim1.copy_from_initial()
    assert stim1 != stim2
    assert np.allclose(stim1._v, stim2.vertices)
    assert np.allclose(stim1._v, stim2._v)
    assert not np.allclose(stim1.vertices, stim2._v)
    assert not np.allclose(stim1.vertices, stim2.vertices)


def test_plot():
    stim = util.make_stim()
    fig, ax = plt.subplots()
    stim.plot(ax)
    plt.close('all')
