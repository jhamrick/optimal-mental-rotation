import numpy as np

from .base import BaseModel


class HillClimbingModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(HillClimbingModel, self).__init__(*args, **kwargs)
        self.direction = None

    ##################################################################
    # Overwritten PyMC sampling methods

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=360, verbose=verbose)

        if self._current_iter == self._iter: # pragma: no cover
            raise RuntimeError(
                "exhausted all iterations, this shouldn't have happened!")

    def draw(self):
        R = self.model['R'].value
        log_S = self.model['log_S'].logp

        if self.direction is None:
            self.direction = np.random.choice([1, -1])
            step = self.direction * np.radians(10)

            self.model['R'].value = R + step
            new_log_S = self.model['log_S'].logp
            if new_log_S < log_S or np.allclose(new_log_S, log_S):
                self.tally()
                self.model['R'].value = R
                self.direction *= -1

            else: # pragma: no cover
                pass

        else:
            step = self.direction * np.radians(10)

            self.model['R'].value = R + step
            new_log_S = self.model['log_S'].logp
            if new_log_S < log_S or np.allclose(new_log_S, log_S):
                self.status = 'halt'

            else: # pragma: no cover
                pass

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
        R = np.linspace(0, 2 * np.pi, 360)
        dZ_dR = self.dZ_dR(R)
        Z = np.trapz(dZ_dR, R)
        return Z

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        Ri = self.R_i
        Si = self.S_i
        R = np.linspace(0, 2 * np.pi, 360)
        S = self.S(R)
        self._plot(
            ax, None, None, Ri, Si, R, S, None,
            title="Linear interpolation for $S$")
