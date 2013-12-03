import numpy as np
from . import Model
from search import hill_climbing


class NaiveModel(Model):

    def __init__(self, *args, **kwargs):
        super(NaiveModel, self).__init__(*args, **kwargs)
        self.Ri = np.append(self.Ri, 2*np.pi)
        self.Si = np.append(self.Si, self.Si[0])

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
        self.S_mean = np.interp(self.R, self.Ri[ix], self.Si[ix])
        self.S_var = np.zeros(self.S_mean.shape)
