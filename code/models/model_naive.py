import numpy as np
from . import Model
from search import hill_climbing


class NaiveModel(Model):

    def next(self):
        """Sample the next point."""

        self.debug("Finding next sample")

        cont = hill_climbing(self)
        self.fit()
        self.integrate()
        self.print_Z(level=0)

        if not cont:
            raise StopIteration

    def fit(self):
        """Fit the likelihood function."""

        self.debug("Fitting likelihood")

        # the samples need to be in sorted order
        ix = np.argsort(self.Ri)
        Ri = np.append(self.Ri[ix], 2*np.pi)
        Si = np.append(self.Si[ix], self.Si[0])

        self.S_mean = np.interp(self.R, Ri, Si)
        self.S_var = np.zeros(self.S_mean.shape)
