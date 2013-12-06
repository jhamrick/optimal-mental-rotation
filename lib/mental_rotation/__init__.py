from path import path
ROOT_PATH = path(__path__[0]).joinpath("../../").abspath()
del path


def get_config(name='config.ini'):
    from ConfigParser import SafeConfigParser
    config = SafeConfigParser()
    config.read(ROOT_PATH.joinpath(name))
    return config

config = get_config()


def get_path(name):
    return ROOT_PATH.joinpath(config.get("paths", name))

STIM_PATH = get_path("stimuli")
DATA_PATH = get_path("data")
FIG_PATH = get_path("figures")
BIN_PATH = get_path("bin")
EXP_PATH = get_path("experiment")

import logging
LOGLEVEL = config.get("global", "loglevel").upper()
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
logging.basicConfig(level=LOGLEVEL, format=FORMAT)

import numpy as np
DTYPE = np.dtype(config.get("global", "dtype"))
del np

from .stimulus import Stimulus2D
