import numpy as np
from numpy.distutils.system_info import get_info
blas_include = get_info('blas_opt')['extra_compile_args'][1][2:]
import pyximport; pyximport.install(
    setup_args={'include_dirs': [blas_include, np.get_include()]})
import model_c

from .base_model import BaseModel
from .gold_standard_model import GoldStandardModel
from .hill_climbing_model import HillClimbingModel
from .bayesian_quadrature_model import BayesianQuadratureModel

__all__ = ['BaseModel', 'GoldStandardModel', 'HillClimbingModel',
           'BayesianQuadratureModel']
