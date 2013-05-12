import numpy as np

from snippets.stats import GP
from model_base import Model
from kernel import KernelMLL
from search import hill_climbing


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
        ntry : int (default=10)
            Number of optimizations to run
        obs_noise : bool (default=False)
            Whether to fit parameter for observation noise

        """

        # default options
        self.opt = {
            'ntry': 10,
            'kernel': 'periodic',
            's': 0,
        }

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)
        self._icurr = 0
        self._ilast = None

        # marginal log likelihood objects
        self._mll_S = KernelMLL(
            kernel=self.opt['kernel'],
            h=self.opt['scale'],
            w=None,
            s=self.opt['s']
        )
        self._mll_logS = KernelMLL(
            kernel=self.opt['kernel'],
            h=np.log(self.opt['scale']+1),
            w=None,
            s=self.opt['s']
        )
        self._mll_Dc = KernelMLL(
            kernel=self.opt['kernel'],
            h=None,
            w=None,
            s=self.opt['s']
        )

    def next(self):
        """Sample the next point."""

        self.debug("Finding next sample")

        icurr = hill_climbing(self)
        if icurr is None:
            raise StopIteration

        self._ilast = self._icurr
        self._icurr = icurr

    def _fit_gp(self, Ri, Si, mll, name):
        self.debug("Fitting parameters for GP over %s ..." % name, level=2)

        # fit parameters
        theta = mll.maximize(
            Ri, Si,
            wmax=np.pi/2.,
            ntry=self.opt['ntry'],
            verbose=self.opt['verbose'] > 3)

        self.debug("Best parameters: %s" % (theta,), level=2)
        self.debug("Computing GP over %s..." % name, level=2)

        # GP regression
        mu, cov = GP(
            mll.kernel(*theta[:-1]), Ri, Si, self.R)

        return mu, cov, theta

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

        # compute GP regressions for S and log(S)
        self.mu_S, self.cov_S, self.theta_S = self._fit_gp(
            self.Ri, self.Si, self._mll_S, "S")
        if ((self.mu_S + 1) <= 0).any():
            print "Warning: regression for mu_S returned negative values"
        self.mu_logS, self.cov_logS, self.theta_logS = self._fit_gp(
            self.Ri, np.log(self.Si + 1), self._mll_logS, "log(S)")

        # choose "candidate" points, halfway between given points
        cix = np.sort(np.unique(np.concatenate([
            (np.array(self.ix) + np.array(self.ix[1:] + [self.R.size])) / 2,
            self.ix])))
        self.delta = self.mu_logS - np.log(self.mu_S + 1)
        self.Rc = self.R[cix].copy()
        self.Dc = self.delta[cix].copy()
        # handle if some of the ycs are nan
        if np.isnan(self.Dc).any():
            goodidx = ~np.isnan(self.Dc)
            self.Rc = self.Rc[goodidx]
            self.Dc = self.Dc[goodidx]

        # compute GP regression for Delta_c -- just use logS parameters
        self.mu_Dc, self.cov_Dc, self.theta_Dc = self._fit_gp(
            self.Rc, self.Dc, self._mll_Dc, "Delta_c")

        # the final regression for S
        #
        # According to the Osborne paper, it should be this:
        #   self.S_mean = ((self.mu_S + 1) * (1 + self.mu_Dc)) - 1
        #
        # But then if self.mu_S < 0, self.S_mean will also have
        # negative parts. To get around this, I am replacing
        # (self.mu_S + 1) with exp(self.mu_logS - self.mu_Dc)
        mu_S_plus_one = np.exp(self.mu_logS - self.mu_Dc)
        self.S_mean = (mu_S_plus_one * (1 + self.mu_Dc)) - 1
        self.S_var = np.diag(self.cov_logS)

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

        # mean
        self.Z_mean = np.sum(self.pR * self.S_mean)

        # variance
        pRmuS = self.pR * (self.mu_S + 1)
        self.Z_var = np.dot(pRmuS, np.dot(self.cov_logS, pRmuS))

        self.print_Z(level=0)
