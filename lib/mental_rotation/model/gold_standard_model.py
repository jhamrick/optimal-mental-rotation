import numpy as np
import logging

from . import BaseModel

logger = logging.getLogger("mental_rotation.model.gs")


class GoldStandardModel(BaseModel):

    _iter = 722

    def draw(self):
        if self._current_iter == 0:
            self.model['R'].value = -np.pi
            self.model['F'].value = 0
            return

        if self._current_iter % 2 == 0:
            self.model['F'].value = 0
            self.model['R'].value = self.model['R'].value + np.radians(1)

        else:
            self.model['F'].value = 1

    ##################################################################
    # The estimated S function

    def log_S(self, R, F):
        return np.log(self.S(R, F))

    def S(self, R, F):
        R_i = self.R_i
        F_i = self.F_i
        match = F_i == F
        ix = np.argsort(R_i[match])
        Ri = R_i[match][ix]
        Si = self.S_i[match][ix]
        S = np.interp(self._unwrap(R), Ri, Si)
        return S

    ##################################################################
    # Plotting methods

    def plot(self, ax, F, f_S=None, color0=None, color='k'):
        if f_S is not None:
            logger.warn("f_S is not used by this function")
        if color0 is not None:
            logger.warn("color0 is not used by this function")

        R = np.linspace(-np.pi, np.pi, 1000)
        S = self.S(R, F)

        lines = {}
        lines['approx'] = ax.plot(R, S, '-', color=color, lw=2)
        return lines
