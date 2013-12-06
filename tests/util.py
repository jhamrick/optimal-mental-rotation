import numpy as np

from mental_rotation.stimulus import Stimulus2D
from mental_rotation import config

R_mu = config.getfloat("model", "R_mu")
R_kappa = config.getfloat("model", "R_kappa")
S_sigma = config.getfloat("model", "S_sigma")


def seed():
    np.random.seed(23480)


def make_stim():
    seed()
    stim = Stimulus2D.random(8)
    return stim


def make_model(cls, R, flip):
    X = Stimulus2D.random(8)
    if flip:
        X.flip([0, 1])
    if R != 0:
        X.rotate(R)
    Xa = X.copy_from_initial()
    Xb = X.copy_from_vertices()
    m = cls(Xa, Xb, R_mu, R_kappa, S_sigma)
    return Xa, Xb, m