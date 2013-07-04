import numpy as np
from . import Model


class GoldStandardModel(Model):

    def __init__(self, *args, **kwargs):
        super(GoldStandardModel, self).__init__(*args, **kwargs)

    def next(self):
        """Sample the next point."""

        if self.num_samples_left == 0:
            raise StopIteration

        rcurr, scurr = self.curr_val
        self.sample(np.degrees(rcurr) + 1)

    def fit(self):
        """Fit the likelihood function."""

        self.S_mean = self.Si.copy()
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
