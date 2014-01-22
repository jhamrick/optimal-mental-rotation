import numpy as np
import matplotlib.pyplot as plt
import pytest
from path import path
from tempfile import NamedTemporaryFile
from mental_rotation import Stimulus2D


def test_vertices_ndim():
    vertices = np.random.rand(8)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_shape():
    vertices = np.random.rand(8, 3)
    with pytest.raises(ValueError):
        Stimulus2D(vertices)


def test_vertices_copy():
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


def test_rotate(X0):
    X0.rotate(90)
    assert X0.operations == [["rotate", 90.0]]
    assert (X0.vertices != X0._v).any()


def test_rotate4(X0):
    X0.rotate(90)
    X0.rotate(90)
    X0.rotate(90)
    X0.rotate(90)
    assert X0.operations == [["rotate", 90.0]]*4
    assert np.allclose(X0.vertices, X0._v)


def test_flip2(X0):
    X0.flip([1, 0])
    X0.flip([1, 0])
    assert X0.operations == [["flip", [1.0, 0.0]]]*2
    assert np.allclose(X0.vertices, X0._v)


def test_equality(X0):
    X1 = Stimulus2D(X0.vertices.copy(), sort=False)
    X2 = Stimulus2D(X0.vertices.copy(), sort=False)

    assert X0 == X0
    assert X1 == X1
    assert X0 == X1
    assert not (X0 != X1)

    X1.rotate(90)
    assert X0 != X1
    v = X1.vertices
    X1.rotate(90)
    X1.rotate(90)
    X1.rotate(90)
    assert X0 != X1

    assert X0 == X2
    X2._v[:] = v
    assert X0 != X2


def test_save(X0):
    fh = NamedTemporaryFile()
    with pytest.raises(IOError):
        X0.save(fh.name, force=False)
    X0.save(fh.name, force=True)


def test_io(X0):
    fh = NamedTemporaryFile()
    X0.save(fh.name, force=True)
    X1 = Stimulus2D.load(fh.name)
    assert X0 == X1


def test_copy_from_state(X0):
    X0.rotate(90)
    X1 = X0.copy_from_state()
    assert X0 == X1
    assert X0 is not X1
    X0.rotate(90)
    assert X0 != X1


def test_copy_from_vertices(X0):
    X0.rotate(90)
    X1 = X0.copy_from_vertices()
    assert X0 != X1
    assert np.allclose(X0.vertices, X1.vertices)
    assert np.allclose(X0.vertices, X1._v)
    assert not np.allclose(X0._v, X1.vertices)
    assert not np.allclose(X0._v, X1._v)


def test_copy_from_initial(X0):
    X0.rotate(90)
    X1 = X0.copy_from_initial()
    assert X0 != X1
    assert np.allclose(X0._v, X1.vertices)
    assert np.allclose(X0._v, X1._v)
    assert not np.allclose(X0.vertices, X1._v)
    assert not np.allclose(X0.vertices, X1.vertices)


def test_plot(X0):
    fig, ax = plt.subplots()
    X0.plot(ax)
    plt.close('all')
