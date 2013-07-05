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

        mu, cov = self.opt['prior_R']
        mu = np.array(mu) * np.ones(1)
        cov = np.array(cov) * np.ones((1, 1))
        self.opt['prior_R'] = mu, cov

        if self.opt['kernel'] == 'gaussian':
            self.Ri = np.append(self.Ri, 2*np.pi)
            self.Si = np.append(self.Si, self.Si[0])

    def next(self):
        """Sample the next point."""

        self.debug("Performing ratio test")

        # check if we can accept a hypothesis
        if self.S_mean is None:
            self.fit()
            self.integrate()
        if self.ratio_test(level=1) > -1:
            raise StopIteration

        self.debug("Finding next sample")

        if self.num_samples_left == 0:
            self.debug("Exhausted all samples", level=2)
            raise StopIteration

        rcurr, scurr = self.curr_val
        d = int(np.round(np.degrees(rcurr)))
        self.debug("Current: S(%d) = %f" % (d, scurr), level=1)

        dr = self.opt['dr']
        w = float(self._bq.gp_S.params[1])
        box = np.max([dr, w])
        r1 = np.radians(1)
        R = np.arange(rcurr-box, rcurr+box+r1, r1)[:, None, None]
        S = np.array([self._bq.expected_uncertainty_evidence(r)
                      if not self.observed(r) else np.nan
                      for r in R])

        ix = np.nanargmin(S)
        self.sample(R[ix])

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

        self._bq = bq.BQ(
            self.R[:, None], self.Ri[:, None], self.Si[:, None], self.opt)
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
