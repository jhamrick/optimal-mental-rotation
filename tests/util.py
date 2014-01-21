import numpy as np
from ConfigParser import RawConfigParser

from mental_rotation.stimulus import Stimulus2D


config = RawConfigParser()
config.add_section('global')
config.set('global', 'loglevel', 'INFO')
config.set('global', 'version', 'test')
config.set('global', 'seed', '23480')

config.add_section('paths')
config.set('paths', 'stimuli', 'stimuli')
config.set('paths', 'data', 'data')
config.set('paths', 'figures', 'figures')
config.set('paths', 'experiment', 'experiment')
config.set('paths', 'simulations', 'data/sim-raw')
config.set('paths', 'resources', 'resources')
config.set('paths', 'sim_scripts', 'resources/sim-scripts')

config.add_section('model')
config.set('model', 'S_sigma', '0.15')

config.add_section('bq')
config.set('bq', 'R_mu', '3.141592653589793')
config.set('bq', 'R_kappa', '0.01')
config.set('bq', 'n_candidate', '20')


def seed():
    seed = config.getint('global', 'seed')
    np.random.seed(seed)


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

    opts = {
        'S_sigma': config.getfloat('model', 'S_sigma')
    }

    m = cls(Xa.vertices, Xb.vertices, **opts)
    return Xa, Xb, m
