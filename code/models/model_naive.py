import numpy as np
from functools import wraps

from . import Model
from search import hill_climbing


class NaiveModel(Model):

    @wraps(Model.__init__)
    def __init__(self, *args, **kwargs):
        super(NaiveModel, self).__init__(*args, **kwargs)
        self._icurr = 0
        self._ilast = None
        self.sample(2*np.pi)

    def next(self):
        """Sample the next point."""

        self.debug("Finding next sample")

        icurr = hill_climbing(self)
        if icurr is None:
            raise StopIteration

        self._ilast = self._icurr
        self._icurr = icurr

    def fit(self):
        """Fit the likelihood function."""

        self.debug("Fitting likelihood")

        # the samples need to be in sorted order
        ix = np.argsort(self.Ri)
        Ri = self.Ri[ix]
        Si = self.Si[ix]

        self.S_mean = np.interp(self.R, Ri, Si)
        self.S_var = np.zeros(self.S_mean.shape)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        self.Z_mean = np.trapz(self.opt['prior_R'] * self.S_mean, self.R)
        self.Z_var = 0
        self.print_Z(level=0)
