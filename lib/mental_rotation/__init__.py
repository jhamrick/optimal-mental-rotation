# configure logging
import logging
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("mental_rotation")

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
