from path import path

# first define the root path
ROOT_PATH = path(__path__[0]).joinpath("../../").abspath()
del path


# load the configuration
def get_config(name='config.ini'):
    from ConfigParser import SafeConfigParser
    config = SafeConfigParser()
    config.read(ROOT_PATH.joinpath(name))
    return config

config = get_config()

VERSION = config.get("global", "version")
SEED = config.getint("global", "seed")


# load the various paths that we need
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

# configure logging
import logging
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
LOGLEVEL = config.get("global", "loglevel").upper()
logging.basicConfig(format=FORMAT, level=LOGLEVEL)
logger = logging.getLogger("mental_rotation")
logger.setLevel(LOGLEVEL)

# import stimulus and model code
from .stimulus import Stimulus2D
from . import model

# make a list of the available models
MODELS = model.__all__[:]
if 'BaseModel' in MODELS:
    MODELS.remove('BaseModel')
MODELS = tuple(MODELS)

# import the rest of the code
from . import sims
