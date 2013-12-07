import numpy as np

from .base import BaseModel


class GoldStandardModel(BaseModel):

    def draw(self):
        self.model['R'].value = self.model['R'].value + np.radians(1)

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=360, verbose=verbose)

    def integrate(self):
        R = np.linspace(0, 2*np.pi, 360)
        p = self.p(R)
        Z = np.trapz(p, R)
        return Z

    def S(self, R):
        Ri = self.R_i
        Si = self.S_i
        ix = np.argsort(Ri)
        S = np.interp(R, Ri[ix], Si[ix])
        return S

    def p(self, R):
        Ri = self.R_i
        pi = self.p_i
        ix = np.argsort(Ri)
        p = np.interp(R, Ri[ix], pi[ix])
        return p

    def plot(self, ax):
        R = np.linspace(0, 2*np.pi, 360)
        S = self.S(R)
        self._plot(
            ax, R, S, None, None, None, None, None,
            title="Likelihood function",
            legend=False)
