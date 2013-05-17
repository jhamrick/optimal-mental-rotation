import numpy as np
from functools import wraps
from model_base import Model


class GoldStandardModel(Model):

    @wraps(Model.__init__)
    def __init__(self, *args, **kwargs):
        super(GoldStandardModel, self).__init__(*args, **kwargs)
        self._rotations = np.arange(self.R.size, dtype='i8')

    def next(self):
        """Sample the next point."""

        self.ix = range(self._rotations.size)
        raise StopIteration

    def fit(self):
        """Fit the likelihood function."""

        self.Ri = self.R[self.ix]
        self.Si = self.S[self.ix]
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


if __name__ == "__main__":
    import util
    opt = util.load_opt()
    stims = util.find_stims()[:opt['nstim']]
    util.run_model(stims, GoldStandardModel, opt)
