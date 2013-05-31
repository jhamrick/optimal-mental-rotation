import numpy as np

from snippets.stats import GP
from model_base import Model
from kernel import KernelMLL
from util import log_clip, safe_multiply


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
        s : float (default=0)
            Observation noise parameter value (if None, will be fit)
        h : float (default=None)
            Output scale parameter (if None, will be fit)
        p : float (default=1)
            Periodic scale/wavelength (if None, will be fit)

        """

        # default options
        self.opt = {
            'ntry': 10,
            'kernel': 'periodic',
            'h': None,
            's': 0,
            'p': 1,
        }

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)
        self._icurr = 0
        self._ilast = None

        # marginal log likelihood objects
        self._mll_S = KernelMLL(
            kernel=self.opt['kernel'],
            h=self.opt['h'],
            w=None,
            s=self.opt['s'],
            p=self.opt['p'],
        )
        self._mll_logS = KernelMLL(
            kernel=self.opt['kernel'],
            h=np.log(self.opt['h'] + 1) if self.opt.get('h', None) else None,
            w=None,
            s=self.opt['s'],
            p=self.opt['p'],
        )
        self._mll_Dc = KernelMLL(
            kernel=self.opt['kernel'],
            h=None,
            w=None,
            s=self.opt['s'],
            p=self.opt['p'],
        )

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
            ntry=self.opt['ntry'],
            verbose=self.opt['verbose'] > 3,
            wmin=np.radians(self.opt['bq_wmin']),
            wmax=np.radians(self.opt['bq_wmax']))

        self.debug("Best parameters: %s" % (theta,), level=2)
        self.debug("Computing GP over %s..." % name, level=2)

        # GP regression
        kernel = mll.make_kernel(params=theta)
        mu, cov = GP(kernel, Ri, Si, self.R)

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
        lSi = np.log(self.Si + 1)

        # compute GP regressions for S and log(S)
        self.mu_S, self.cov_S, self.theta_S = self._fit_gp(
            self.Ri, self.Si, self._mll_S, "S")
        if ((self.mu_S + 1) <= 0).any():
            print "Warning: regression for mu_S returned negative values"
        self.mu_logS, self.cov_logS, self.theta_logS = self._fit_gp(
            self.Ri, lSi, self._mll_logS, "log(S)")

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

        # the final estimated mean of S
        self.S_mean = ((self.mu_S + 1) * (1 + self.mu_Dc)) - 1

        # marginalize out w
        params = self._mll_logS.free_params(self.theta_logS)
        if self._mll_logS.h is None:
            ii = 1
        else:
            ii = 0
        # estimate the variance
        Hw = self._mll_logS.hessian(params, self.Ri, lSi)
        Cw = np.matrix(np.exp(log_clip(-np.diag(Hw))[ii]))
        dm_dw = np.matrix(
            self._mll_logS.dm_dw(params, self.Ri, lSi, self.R))
        self.S_cov = np.array(
            self.cov_logS + np.dot(dm_dw.T, np.dot(Cw, dm_dw)))
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

        # mean
        self.Z_mean = np.trapz(self.opt['prior_R'] * self.S_mean, self.R)

        # variance
        pm = self.opt['prior_R'] * (self.mu_S + 1)
        C = safe_multiply(self.S_cov, pm[:, None], pm[None, :])
        C[np.abs(C) < 1e-100] = 0
        upper = np.trapz(np.trapz(C, self.R, axis=0), self.R)

        m = self.mu_S
        m[m < 0] = 0
        pm = self.opt['prior_R'] * m
        C = safe_multiply(self.S_cov, pm[:, None], pm[None, :])
        C[np.abs(C) < 1e-100] = 0
        lower = np.trapz(np.trapz(C, self.R, axis=0), self.R)

        self.Z_var = (lower, upper)
        self.print_Z(level=0)


if __name__ == "__main__":
    import util
    import sys

    # load options
    opt = util.load_opt()

    # run each stim
    util.run_all(sys.argv[1:], BayesianQuadratureModel, opt)
