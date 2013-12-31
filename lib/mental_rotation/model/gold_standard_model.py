import numpy as np

from . import BaseModel


class GoldStandardModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(GoldStandardModel, self).__init__(*args, **kwargs)

    ##################################################################
    # Overwritten PyMC sampling methods

    def sample(self, verbose=0):
        self.model['R'].value = -np.pi
        super(BaseModel, self).sample(iter=361, verbose=verbose)

    def draw(self):
        self.model['R'].value = self.model['R'].value + np.radians(1)

    ##################################################################
    # The estimated S function

    def log_S(self, R):
        return np.log(self.S(R))

    def S(self, R):
        ix = np.argsort(self.R_i)
        Ri = self.R_i[ix]
        Si = self.S_i[ix]
        S = np.interp(self._unwrap(R), Ri, Si)
        return S

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    def log_dZ_dR(self, R):
        return np.log(self.dZ_dR(R))

    def dZ_dR(self, R):
        ix = np.argsort(self.R_i)
        Ri = self.R_i[ix]
        dZ_dRi = self.dZ_dR_i[ix]
        dZ_dR = np.interp(self._unwrap(R), Ri, dZ_dRi)
        return dZ_dR

    @property
    def log_Z(self):
        return np.log(self.Z)

    @property
    def Z(self):
        R = np.linspace(-np.pi, np.pi, 361)
        dZ_dR = self.dZ_dR(R)
        Z = np.trapz(dZ_dR, R)
        return Z

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        R = np.linspace(-np.pi, np.pi, 361)
        S = self.S(R)
        self._plot(
            ax, R, S, None, None, None, None, None,
            title="Likelihood function",
            legend=False)
