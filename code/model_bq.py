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
            'obs_noise': False
        }

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)
        self._icurr = 0

        # marginal log likelihood object
        self._mll = KernelMLL(
            kernel=self.opt['kernel'],
            obs_noise=self.opt['obs_noise']
        )

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

    def _fit_gp(self, Ri, Si, name, wmin=1e-8):
        if self.opt['verbose']:
            print "Fitting parameters for GP over %s ..." % name

        # fit parameters
        theta = self._mll.maximize(
            Ri, Si,
            wmin=wmin,
            ntry=self.opt['ntry'],
            verbose=self.opt['verbose'])

        if self.opt['verbose']:
            print theta
            print "Computing GP over %s..." % name

        # GP regression
        mu, cov = GP(
            self._mll.kernel(*theta), Ri, Si, self.R)

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

        # input data
        self.ix = sorted(self.ix)
        self.Ri = self.R[self.ix[:-1]].copy()
        self.Si = self.S[self.ix[:-1]].copy()

        # compute GP regressions for S and log(S)
        self.mu_S, self.cov_S, self.theta_S = self._fit_gp(
            self.Ri, self.Si, "S")
        if ((self.mu_S + 1) <= 0).any():
            print "Warning: regression for mu_S returned negative values"
        self.mu_logS, self.cov_logS, self.theta_logS = self._fit_gp(
            self.Ri, np.log(self.Si + 1), "log(S)")

        # choose "candidate" points, halfway between given points
        cix = cix = np.sort(np.unique(np.concatenate([
            (np.array(self.ix)[:-1] + np.array(self.ix)[1:]) / 2,
            self.ix[:-1]])))
        self.delta = self.mu_logS - np.log(self.mu_S + 1)
        self.Rc = self.R[cix].copy()
        self.Sc = self.delta[cix].copy()
        # handle if some of the ycs are nan
        if np.isnan(self.Sc).any():
            goodidx = ~np.isnan(self.Sc)
            self.Rc = self.Rc[goodidx]
            self.Sc = self.Sc[goodidx]

        # compute GP regression for Delta_c -- just use logS parameters
        self.mu_Dc, self.cov_Dc = GP(
            self._mll.kernel(*self.theta_logS),
            self.Rc, self.Sc, self.R)

        # the final regression for S
        self.S_mean = ((self.mu_S + 1) * (1 + self.mu_Dc)) - 1
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

        if self.opt['verbose']:
            print "Computing mean and variance of estimate of Z..."

        # mean
        self.Z_mean = np.sum(self.pR * self.S_mean)

        # variance
        pRmuS = self.pR * self.mu_S
        self.Z_var = np.dot(pRmuS, np.dot(self.cov_logS, pRmuS))
