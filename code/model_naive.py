import numpy as np

from model_base import Model
from search import hill_climbing


class NaiveModel(Model):

    def __init__(self, *args, **kwargs):
        super(NaiveModel, self).__init__(*args, **kwargs)
        self._icurr = 0
        self._ilast = None

    def next(self):
        """Sample the next point."""

        icurr = hill_climbing(self)
        if icurr is None:
            raise StopIteration

        self._ilast = self._icurr
        self._icurr = icurr

    def fit(self):
        """Fit the likelihood function."""

        if self.opt['verbose']:
            print "Fitting likelihood..."

        self.ix = sorted(self.ix) + [0]
        self.Ri = self.R[self.ix]
        self.Ri[-1] = 2*np.pi
        self.Si = self.S[self.ix]
        self.S_mean = np.interp(self.R, self.Ri, self.Si)
        self.S_var = np.zeros(self.S_mean.shape)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        if self.opt['verbose']:
            print "Computing mean and variance of estimate of Z..."

        self.Z_mean = sum(self.pR * self.S_mean)
        self.Z_var = 0
