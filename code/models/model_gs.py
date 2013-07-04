import numpy as np
import scipy
from . import Model


class GoldStandardModel(Model):

    def __init__(self, *args, **kwargs):
        super(GoldStandardModel, self).__init__(*args, **kwargs)

    def next(self):
        """Sample the next point."""

        while self.num_samples_left > 0:
            rcurr, scurr = self.curr_val
            self.sample(rcurr + np.radians(1))

        self.fit()
        self.integrate()
        raise StopIteration

    def fit(self):
        """Fit the likelihood function."""

        ix = np.argsort(self.Ri)
        self.S_mean = np.append(self.Si[ix], self.Si[0])
        self.S_var = np.zeros(self.S_mean.shape)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        mu, var = self.opt['prior_R']
        p = scipy.stats.norm.pdf(self.R, mu, np.sqrt(var))
        self.Z_mean = np.trapz(p * self.S_mean, self.R)
        self.Z_var = 0
