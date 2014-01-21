import numpy as np
from path import path
from ConfigParser import RawConfigParser

from mental_rotation.stimulus import Stimulus2D


def setup_config(tmp_path):
    config = RawConfigParser()
    config.add_section('global')
    config.set('global', 'loglevel', 'INFO')
    config.set('global', 'version', 'test')
    config.set('global', 'seed', '23480')

    config.add_section('paths')
    config.set('paths', 'stimuli', 'tests/stimuli')
    config.set('paths', 'data', tmp_path.joinpath('data'))
    config.set('paths', 'figures', tmp_path.joinpath('figures'))
    config.set('paths', 'experiment', tmp_path.joinpath('experiment'))
    config.set('paths', 'simulations', tmp_path.joinpath('data/sim-raw'))
    config.set('paths', 'resources', tmp_path.joinpath('resources'))

    config.add_section('experiment')
    config.set('experiment', 'deploy_path', tmp_path.joinpath('deploy'))
    config.set('experiment', 'fetch_path', 'http://localhost:22361/data')
    
    config.add_section('model')
    config.set('model', 'S_sigma', '0.15')
    
    config.add_section('bq')
    config.set('bq', 'R_mu', '3.141592653589793')
    config.set('bq', 'R_kappa', '0.01')
    config.set('bq', 'n_candidate', '20')

    return config


def make_stim():
    stim = Stimulus2D.random(8)
    return stim


def make_circle():
    R = np.radians(np.arange(0, 360, 10))
    v = np.empty((R.size, 2))
    v[:, 0] = np.cos(R)
    v[:, 1] = np.sin(R)
    X = Stimulus2D(v)
    return X
