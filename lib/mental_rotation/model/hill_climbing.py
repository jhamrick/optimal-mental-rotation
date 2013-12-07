import numpy as np

from .base import BaseModel


class HillClimbingModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(HillClimbingModel, self).__init__(*args, **kwargs)
        self.direction = None

    def draw(self):
        R = self.model['R'].value
        logS = self.model['logS'].logp

        if self.direction is None:
            self.direction = np.random.choice([1, -1])
            step = self.direction * np.radians(10)

            self.model['R'].value = R + step
            new_logS = self.model['logS'].logp
            if new_logS < logS or np.allclose(new_logS, logS):
                self.tally()
                self.model['R'].value = R
                self.direction *= -1

            else: # pragma: no cover
                pass

        else:
            step = self.direction * np.radians(10)

            self.model['R'].value = R + step
            new_logS = self.model['logS'].logp
            if new_logS < logS or np.allclose(new_logS, logS):
                self.status = 'halt'

            else: # pragma: no cover
                pass

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=360, verbose=verbose)

        if self._current_iter == self._iter: # pragma: no cover
            raise RuntimeError(
                "exhausted all iterations, this shouldn't have happened!")

    def integrate(self):
        R = np.linspace(0, 2 * np.pi, 360)
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
        Ri = self.R_i
        Si = self.S_i
        R = np.linspace(0, 2 * np.pi, 360)
        S = self.S(R)
        self._plot(
            ax, None, None, Ri, Si, R, S, None,
            title="Linear interpolation for $S$")
