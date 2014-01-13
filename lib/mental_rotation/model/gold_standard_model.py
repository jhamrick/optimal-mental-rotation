import numpy as np

from . import BaseModel


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
    # Estimated dZ_dR and full estimate of Z

    def log_dZ_dR(self, R, F):
        return np.log(self.dZ_dR(R, F))

    def dZ_dR(self, R, F):
        R_i = self.R_i
        F_i = self.F_i
        match = F_i == F
        ix = np.argsort(R_i[match])
        Ri = R_i[match][ix]
        dZ_dRi = self.dZ_dR_i[match][ix]
        dZ_dR = np.interp(self._unwrap(R), Ri, dZ_dRi)
        return dZ_dR

    def log_Z(self, F):
        return np.log(self.Z(F))

    def Z(self, F):
        R = np.linspace(-np.pi, np.pi, 361)
        dZ_dR = self.dZ_dR(R, F)
        Z = np.trapz(dZ_dR, R)
        return Z

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        R = np.linspace(-np.pi, np.pi, 361)
        S0 = self.S(R, F=0)
        S1 = self.S(R, F=1)
        ax.plot(R, S0, '-', color="#550000", label="Truth, F=0", lw=2)
        ax.plot(R, S1, '-', color="#000055", label="Truth, F=1", lw=2)
        ax.legend()
