import numpy as np

from model_base import Model as BaseModel
from model_gs import GoldStandardModel as GoldStandard
from model_naive import NaiveModel as Naive
from model_vm import VonMisesModel as VonMises
from model_bq import BayesianQuadratureModel as BayesianQuadrature

prior_X = BaseModel.prior_X
