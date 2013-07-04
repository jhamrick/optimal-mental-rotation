import numpy as np

from . import Model
import bq
reload(bq)


class BayesianQuadratureModel(Model):
    """Estimate a likelihood function, S(y|x) using Gaussian Process
    regressions, as in Osborne et al. (2012):

    1) Estimate S using a GP
    2) Estimate log(S) using second GP
    3) Estimate delta_C using a third GP

    References
    ----------
    Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
        Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
        Model Evidence Using Bayesian Quadrature. *Advances in Neural
        Information Processing Systems*, 25.

    """

    def __init__(self, *args, **kwargs):
        """See Model.__init__

        Additional model options
        ------------------------
        kernel : string
            Type of kernel to use, either 'gaussian' or 'periodic'
        ntry_bq : int (default=10)
            Number of optimizations to run
        s : float (default=0)
            Observation noise parameter value (if None, will be fit)
        h : float (default=None)
            Output scale parameter (if None, will be fit)
        p : float (default=1)
            Periodic scale/wavelength (if None, will be fit)

        """

        # default options
        self.opt = {
            'ntry_bq': 10,
            'kernel': 'periodic',
            'h': None,
            's': 0,
            'p': 1,
            'gamma': 1,
        }

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)
        self._dir = 0
        self._icurr = 0

        mu, cov = self.opt['prior_R']
        mu = np.array(mu) * np.ones(1)
        cov = np.array(cov) * np.ones((1, 1))
        self.opt['prior_R'] = mu, cov

    def next(self):
        """Sample the next point."""

        self.debug("Performing ratio test")

        # check if we can accept a hypothesis
        if self.S_mean is None:
            self.fit()
            self.integrate()
        hyp = self.ratio_test(level=1)
        if hyp != -1:
            raise StopIteration

        self.debug("Finding next sample")

        inext = []
        rnext = []
        rr = list(self._rotations)
        for r in self.ix:
            if r not in rr:
                continue
            i = rr.index(r)
            n = (i + 1) % self._rotations.size
            rn = self._rotations[n]
            p = (i - 1) % self._rotations.size
            rp = self._rotations[p]
            if (rn not in self.ix) and (rn not in inext):
                inext.append(n)
                rnext.append(rn)
            if (rp not in self.ix) and (rp not in inext):
                inext.append(p)
                rnext.append(rp)

        nextvars = []
        nexti = []
        nextr = []

        for r, i in zip(rnext, inext):
            if r in self.ix:
                continue

            xpc_unc = self._bq.expected_uncertainty_evidence(np.array([r]))
            print r, xpc_unc

            nexti.append(i)
            nextr.append(r)
            nextvars.append(xpc_unc)

        nextvars = np.array(nextvars)
        nexti = np.array(nexti)
        nextr = np.array(nextr)

        assert nextvars.size <= 2
        if nextvars.size == 0:
            self.debug("Exhausted all samples", level=2)
            raise StopIteration
        elif nextvars.size == 1:
            idx = 0
            assert self._dir == np.sign(nexti[idx] - self._icurr)
        else:
            idx = np.argmin(nextvars)
            if np.abs(nexti[idx] - self._icurr) == 1:
                self._dir = np.sign(nexti[idx] - self._icurr)
            else:
                if self._dir == 0:
                    assert nexti[idx] in (1, self._rotations.size-1)
                    self._dir = 1 if nexti[idx] == 1 else -1
                else:
                    self._dir = -self._dir

        self.debug("Choosing next: %d" % nextr[idx], level=2)
        self.sample(nextr[idx])
        self._icurr = nexti[idx]

        self.fit()
        self.integrate()

    def fit(self):
        """Run the GP regressions to fit the likelihood function.

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        self.debug("Fitting likelihood")

        # input data
        self.ix = sorted(self.ix)
        self.Ri = self.R[self.ix].copy()
        self.Si = self.S[self.ix].copy()

        self._bq = bq.BQ(self.R, self.S, self.ix, self.opt)
        self.S_mean, self.S_cov = self._bq.fit()
        self.S_var = np.diag(self.S_cov)

    def integrate(self):
        """Compute the mean and variance of Z:

        $$Z = \int S(X_b, X_R)p(R) dR$$

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        if self.S_mean is None or self.S_var is None:
            raise RuntimeError(
                "S_mean or S_var is not set, did you call self.fit first?")

        self.Z_mean = self._bq.mean
        self.Z_var = self._bq.var
        self.print_Z(level=0)
