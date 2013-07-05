import numpy as np
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
