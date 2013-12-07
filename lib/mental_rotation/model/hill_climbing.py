import numpy as np

from .base import BaseModel


class HillClimbingModel(BaseModel):

    def draw(self):
        value = self.model['R'].value
        logp = self.logp

        dir = np.random.choice([1, -1])
        self.model['R'].value = value + dir*np.radians(10)

        if self.logp < logp:
            self.tally()
            self.model['R'].value = value - dir*np.radians(10)

            if self.logp < logp:
                self.tally()
                self.model['R'].value = value
                self.status = 'halt'
            else: # pragma: no cover
                pass

        else: # pragma: no cover
            pass

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=360, verbose=verbose)

        if self._current_iter == self._iter: # pragma: no cover
            raise RuntimeError(
                "exhausted all iterations, this shouldn't have happened!")

    def _loop(self):
        self.status = 'running'
        try:
            while not self.status == 'halt':
                if self.status == 'paused': # pragma: no cover
                    break

                self.draw()
                self.tally()

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status = 'halt'

        if self.status == 'halt':
            self._halt()
        else: # pragma: no cover
            pass

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
        Ri = self.R_i
        Si = self.S_i
        R = np.linspace(0, 2*np.pi, 360)
        S = self.S(R)
        self._plot(
            ax, None, None, Ri, Si, R, S, None,
            title="Linear interpolation for $S$")
