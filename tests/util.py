import numpy as np

from mental_rotation.stimulus import Stimulus2D


def seed():
    np.random.seed(23480)


def make_stim():
    seed()
    stim = Stimulus2D.random(8)
    return stim


def make_circle():
    R = np.radians(np.arange(0, 360, 10))
    v = np.empty((R.size, 2))
    v[:, 0] = np.cos(R)
    v[:, 1] = np.sin(R)
    X = Stimulus2D(v)
    return X


def make_model(cls, flip=True, name=None, theta=None):
    X = make_stim()
    if flip:
        X.flip([0, 1])
    if theta is None:
        X.rotate(39)
    else:
        X.rotate(theta)
    Xa = X.copy_from_initial()
    Xb = X.copy_from_vertices()
    m = cls(Xa.vertices, Xb.vertices, name=name)
    return Xa, Xb, m
