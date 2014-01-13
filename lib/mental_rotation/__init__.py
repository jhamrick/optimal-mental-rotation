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
SIM_PATH = get_path("simulations")
RESOURCE_PATH = get_path("resources")
SIM_SCRIPT_PATH = get_path("sim_scripts")

import logging
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
LOGLEVEL = config.get("global", "loglevel").upper()
logging.basicConfig(format=FORMAT, level=LOGLEVEL)

import numpy as np
DTYPE = np.dtype(config.get("global", "dtype"))
del np

SEED = config.getint("global", "seed")

from .stimulus import Stimulus2D
