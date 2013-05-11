import numpy as np
from model_base import Model


class NaiveModel(Model):

    def __init__(self, *args, **kwargs):
        super(NaiveModel, self).__init__(*args, **kwargs)
        self._icurr = 0

    def next(self):
        """Sample the next point."""

        inext = self._icurr + 1
        iprev = self._icurr - 1

        rcurr = self._rotations[self._icurr]
        rnext = self._rotations[inext]
        rprev = self._rotations[iprev]

        scurr = self.sample(rcurr)
        snext = self.sample(rnext)
        sprev = self.sample(rprev)

        if snext > scurr and snext > sprev:
            self._icurr = inext
        elif sprev > scurr and sprev > snext:
            self._icurr = iprev
        else:
            raise StopIteration

    def fit(self):
        """Fit the likelihood function."""

        if self.opt['verbose']:
           print "Fitting likelihood..."

        self.ix = sorted(self.ix)
        self.Ri = self.R[self.ix]
        self.Si = self.S[self.ix]
        self.m_S = np.interp(self.R, self.Ri, self.Si)
        self.V_S = np.zeros(self.m_S.shape)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        """

        if self.m_S is None:
            raise RuntimeError(
                "self.m_S is not set, did you call self.fit first?")

        if self.opt['verbose']:
           print "Computing mean and variance of estimate of Z..."

        self.m_Z = sum(self.pR * self.m_S)
        self.V_Z = 0
