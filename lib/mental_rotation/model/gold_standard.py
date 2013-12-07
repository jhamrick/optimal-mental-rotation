import numpy as np

from .base import BaseModel


class GoldStandardModel(BaseModel):

    ##################################################################
    # Overwritten PyMC sampling methods

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=360, verbose=verbose)

    def draw(self):
        self.model['R'].value = self.model['R'].value + np.radians(1)

    ##################################################################
    # The estimated S function

    def log_S(self, R):
        return np.log(self.S(R))

    def S(self, R):
        Ri = self.R_i
        Si = self.S_i
        ix = np.argsort(Ri)

        sRi = np.empty(Ri.size + 1)
        sRi[:-1] = Ri[ix]
        sRi[-1] = 2 * np.pi

        sSi = np.empty(Si.size + 1)
        sSi[:-1] = Si[ix]
        sSi[-1] = sSi[0]

        S = np.interp(R, sRi, sSi)
        return S

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    def log_dZ_dR(self, R):
        return np.log(self.dZ_dR(R))

    def dZ_dR(self, R):
        Ri = self.R_i
        dZ_dRi = self.dZ_dR_i
        ix = np.argsort(Ri)

        sRi = np.empty(Ri.size + 1)
        sRi[:-1] = Ri[ix]
        sRi[-1] = 2 * np.pi

        sdZ_dRi = np.empty(dZ_dRi.size + 1)
        sdZ_dRi[:-1] = dZ_dRi[ix]
        sdZ_dRi[-1] = sdZ_dRi[0]

        dZ_dR = np.interp(R, sRi, sdZ_dRi)
        return dZ_dR

    @property
    def log_Z(self):
        return np.log(self.Z)

    @property
    def Z(self):
        R = np.linspace(0, 2 * np.pi, 361)
        dZ_dR = self.dZ_dR(R)
        Z = np.trapz(dZ_dR, R)
        return Z

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        R = np.linspace(0, 2 * np.pi, 361)
        S = self.S(R)
        self._plot(
            ax, R, S, None, None, None, None, None,
            title="Likelihood function",
            legend=False)
