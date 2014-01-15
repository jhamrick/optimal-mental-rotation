import numpy as np

from . import BaseModel


class HillClimbingModel(BaseModel):

    _iter = 720

    def __init__(self, *args, **kwargs):
        super(HillClimbingModel, self).__init__(*args, **kwargs)
        self.direction = None

    ##################################################################
    # Sampling

    def draw(self):
        if self._current_iter == 0:
            self.model['R'].value = 0
            self.model['F'].value = 0
            return

        R = self.model['R'].value
        F = self.model['F'].value
        log_S = self.model['log_S'].logp

        if self.direction is None:
            self.direction = np.random.choice([1, -1])
            step = self.direction * self.opts['step']
            self.model['R'].value = R + step

        else:
            step = self.direction * self.opts['step']

            self.model['R'].value = R + step
            new_log_S = self.model['log_S'].logp
            if new_log_S < log_S or np.allclose(new_log_S, log_S):
                if F == 0:
                    self.tally()
                    self._current_iter += 1
                    self.direction = None
                    self.model['R'].value = 0
                    self.model['F'].value = 1
                
                else:
                    self.status = 'done'

            else: # pragma: no cover
                pass

    ##################################################################
    # The estimated S function

    def log_S(self, R, F):
        return np.log(self.S(R, F))

    def S(self, R, F):
        Fi = self.F_i
        match = Fi == F
        
        R_ = self._wrap(R)
        Ri = self._wrap(self.R_i)

        Si = self.S_i
        ix = np.argsort(Ri[match])

        sRi = np.empty(Ri[match].size + 1)
        sRi[:-1] = Ri[match][ix]
        sRi[-1] = 2 * np.pi

        sSi = np.empty(Si[match].size + 1)
        sSi[:-1] = Si[match][ix]
        sSi[-1] = sSi[0]

        S = np.interp(R_, sRi, sSi)
        return S

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    def log_dZ_dR(self, R, F):
        return np.log(self.dZ_dR(R, F))

    def dZ_dR(self, R, F):
        Fi = self.F_i
        match = Fi == F

        R_ = self._wrap(R)
        Ri = self._wrap(self.R_i)

        dZ_dRi = self.dZ_dR_i
        ix = np.argsort(Ri[match])

        sRi = np.empty(Ri[match].size + 1)
        sRi[:-1] = Ri[match][ix]
        sRi[-1] = 2 * np.pi

        sdZ_dRi = np.empty(dZ_dRi[match].size + 1)
        sdZ_dRi[:-1] = dZ_dRi[match][ix]
        sdZ_dRi[-1] = sdZ_dRi[0]

        dZ_dR = np.interp(R_, sRi, sdZ_dRi)
        return dZ_dR

    def log_Z(self, F):
        return np.log(self.Z(F))

    def Z(self, F):
        R = np.linspace(-np.pi, np.pi, 360)
        dZ_dR = self.dZ_dR(R, F)
        Z = np.trapz(dZ_dR, R)
        return Z

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        Fi = self.F_i
        Ri0 = self.R_i[Fi == 0]
        Si0 = self.S_i[Fi == 0]
        Ri1 = self.R_i[Fi == 1]
        Si1 = self.S_i[Fi == 1]
        R = np.linspace(-np.pi, np.pi, 360)
        S0 = self.S(R, F=0)
        S1 = self.S(R, F=1)
        ax.plot(R, S0, 'r-', label="Approx, F=0", lw=2)
        ax.plot(Ri0, Si0, 'ro', markersize=5)
        ax.plot(R, S1, 'b-', label="Approx, F=1", lw=2)
        ax.plot(Ri1, Si1, 'bo', markersize=5)
        ax.legend()

    ##################################################################
    # Copying/Saving

    def __getstate__(self):
        state = super(HillClimbingModel, self).__getstate__()
        state['direction'] = self.direction
        return state

    def __setstate__(self, state):
        super(HillClimbingModel, self).__setstate__(state)
        self.direction = state['direction']
