import numpy as np
import matplotlib.pyplot as plt

from .base import BaseModel


class GoldStandardModel(BaseModel):

    def draw(self):
        for v in self.stochastics:
            if str(v) == "R":
                v.value = v.value + np.radians(1)
            else:
                raise RuntimeError("unhandled variable: %s" % v)

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=359, verbose=verbose)

    def integrate(self):
        R = self.trace('R')[:]
        p = np.exp(self.trace('logp')[:])
        Z = np.trapz(p, R)
        return Z

    def plot(self):
        plt.figure()
        R = self.trace('R')[:]
        S = np.exp(self.trace('S')[:])
        self._plot(
            R, S, None, None, None, None, None,
            title="Likelihood function",
            legend=False)
        plt.ylim(0, 1.45)
