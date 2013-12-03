import numpy as np
import pytest
from tempfile import NamedTemporaryFile
from ..stimulus import Stimulus2D


def seed():
    np.random.seed(23480)


def make_stim():
    seed()
    stim = Stimulus2D.random(8)
    return stim


def test_vertices_ndim():
    seed()
    vertices = np.random.rand(8)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_shape():
    seed()
    vertices = np.random.rand(8, 3)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_copy():
    seed()
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
    stim = make_stim()
    stim.rotate(90)
    assert stim.operations == [["rotate", 90.0]]
    assert (stim.vertices != stim._v).any()


def test_rotate4():
    stim = make_stim()
    stim.rotate(90)
    stim.rotate(90)
    stim.rotate(90)
    stim.rotate(90)
    assert stim.operations == [["rotate", 90.0]]*4
    assert np.allclose(stim.vertices, stim._v)


def test_flip2():
    stim = make_stim()
    stim.flip([1, 0])
    stim.flip([1, 0])
    assert stim.operations == [["flip", [1.0, 0.0]]]*2
    assert np.allclose(stim.vertices, stim._v)


def test_equality():
    stim1 = make_stim()
    stim2 = make_stim()
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

    stim2 = make_stim()
    assert stim1 == stim2
    stim2._v[:] = v
    assert stim1 != stim2


def test_save():
    fh = NamedTemporaryFile()
    stim = make_stim()
    with pytest.raises(IOError):
        stim.save(fh.name, force=False)
    stim.save(fh.name, force=True)


def test_io():
    fh = NamedTemporaryFile()
    stim1 = make_stim()
    stim1.save(fh.name, force=True)
    stim2 = Stimulus2D.load(fh.name)
    assert stim1 == stim2
