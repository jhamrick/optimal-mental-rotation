import numpy as np

from snippets.stats import GP
from model_base import Model
from kernel import KernelMLL


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

        self.fit()
        self.integrate()

    def next(self):
        """Sample the next point."""

        self.debug("Performing ratio test")

        # check if we can accept a hypothesis
        hyp = self.ratio_test(level=1)
        if hyp != -1:
            raise StopIteration

        self.debug("Finding next sample")

        inext = self._icurr + 1
        iprev = self._icurr - 1

        n = self._rotations.size
        if inext >= n or np.abs(iprev) >= n:
            self.debug("Exhausted all samples", level=2)
            raise StopIteration

        rcurr = self._rotations[self._icurr]
        rnext = self._rotations[inext]
        rprev = self._rotations[iprev]

        scurr = self.sample(rcurr)
        snext = self.sample(rnext)
        sprev = self.sample(rprev)

        self.debug("Current value: %f" % scurr, level=2)

        choose_next = False
        choose_prev = False

        if (snext > sprev) and (inext != self._ilast):
            choose_next = True
        elif (sprev > snext) and (iprev != self._ilast):
            choose_prev = True
        elif (self._icurr - self._ilast) > 0:
            choose_next = True
        else:
            choose_prev = True

        if choose_next and not choose_prev:
            self.debug("Choosing next: %d" % inext, level=2)
            icurr = inext
        else:
            self.debug("Choosing prev: %d" % iprev, level=2)
            icurr = iprev

        self._ilast = self._icurr
        self._icurr = icurr

        self.fit()
        self.integrate()

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

    def _candidate(self):
        nc = len(self.ix) * 2
        ideal = list(np.linspace(0, self.R.size, nc+1).astype('i8')[:-1])
        for i in self.ix:
            closest = np.argmin(np.abs(np.array(ideal) - i))
            del ideal[closest]
        c = sorted(ideal + self.ix)
        return c

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
        self.delta = self.mu_logS - np.log(self.mu_S + 1)
        cix = self._candidate()
        self.Rc = self.R[cix].copy()
        self.Dc = self.delta[cix].copy()
        # handle if some of the ycs are nan
        if np.isnan(self.Dc).any():
            goodidx = ~np.isnan(self.Dc)
            self.Rc = self.Rc[goodidx]
            self.Dc = self.Dc[goodidx]

        # compute GP regression for Delta_c -- just use logS parameters
        try:
            self.mu_Dc, self.cov_Dc, self.theta_Dc = self._fit_gp(
                self.Rc, self.Dc, self._mll_Dc, "Delta_c")
        except RuntimeError:
            Rc = list(self.Rc) + [2*np.pi]
            Dc = list(self.Dc) + [self.Rc[0]]
            self.mu_Dc = np.interp(self.R, Rc, Dc)
            self.cov_Dc = np.zeros((self.mu_Dc.size, self.mu_Dc.size))
            self.theta_Dc = None

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
        self.Z_mean = np.trapz(self.opt['prior_R'] * self.S_mean, self.R)

        # variance
        pm = self.opt['prior_R'] * (self.mu_S + 1)
        C = self.cov_logS * pm[:, None] * pm[None, :]
        self.Z_var = np.trapz(np.trapz(C, self.R, axis=0), self.R)

        self.print_Z(level=0)


if __name__ == "__main__":
    import util
    opt = util.load_opt()
    stims = util.find_stims()[:5]
    util.run_model(stims, BayesianQuadratureModel, opt)
