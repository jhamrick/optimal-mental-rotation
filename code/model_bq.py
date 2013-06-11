import numpy as np
import scipy.optimize as optim
from numpy import dot

from model_base import Model

from snippets.safemath import MIN_LOG, EPS
from periodic_kernel import PeriodicKernel as kernel
from gaussian_process import GP

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
        self.log_S = self.log_transform(self.S)
        self._dir = 0
        self._icurr = 0

    def log_transform(self, x):
        return np.log((x / self.opt['gamma']) + 1)

    def E(self, f, axis=-1, p_R=None):
        if p_R is None:
            p_R = self.opt['prior_R']
        pfx = f * p_R
        m = np.trapz(pfx, self.R, axis=axis)
        return m

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

        m = self.gp_logS.m
        C = self.S_cov
        nextvars = []
        nexti = []
        nextr = []

        for r, i in zip(rnext, inext):
            if r in self.ix:
                continue

            Ri = np.concatenate([self.Ri, [self.R[r]]])
            Si = np.concatenate([self.Si, [m[r]]])
            lSi = np.log(Si + 1)

            SK = self._mll_S.make_kernel(params=self.theta_S)
            Smu, Scov = GP(SK, Ri, Si, self.R)

            logSK = self._mll_logS.make_kernel(params=self.theta_logS)
            logSmu, logScov = GP(logSK, Ri, lSi, self.R)

            delta = logSmu - np.log(Smu + 1)
            cix = sorted(set(self._candidate() + [r]))
            Rc = self.R[cix]
            Dc = delta[cix]
            if self.theta_Dc:
                DcK = self._mll_Dc.make_kernel(params=self.theta_Dc)
                try:
                    Dcmu = GP(DcK, Rc, Dc, self.R)[0]
                except np.linalg.LinAlgError:
                    Rc = list(Rc) + [2*np.pi]
                    Dc = list(Dc) + [Rc[0]]
                    Dcmu = np.interp(self.R, Rc, Dc)
            else:
                Rc = list(Rc) + [2*np.pi]
                Dc = list(Dc) + [Rc[0]]
                Dcmu = np.interp(self.R, Rc, Dc)

            Samean = ((Smu + 1) * (1 + Dcmu)) - 1

            params = self._mll_logS.free_params(self.theta_logS)
            if self._mll_logS.h is None:
                ii = 1
            else:
                ii = 0
            Hw = self._mll_logS.hessian(params, Ri, lSi)
            Cw = np.matrix(np.exp(log_clip(-np.diag(Hw))[ii]))
            dm_dw = np.matrix(
                self._mll_logS.dm_dw(params, Ri, lSi, self.R))
            Sacov = np.array(
                logScov + np.dot(dm_dw.T, np.dot(Cw, dm_dw)))

            # Zmean = np.trapz(self.opt['prior_R'] * Samean, self.R)
            # variance
            pm = self.opt['prior_R'] * (Samean + 1)
            C = safe_multiply(Sacov, pm[:, None], pm[None, :])
            C[np.abs(C) < 1e-100] = 0
            Zvar = np.trapz(np.trapz(C, self.R, axis=0), self.R)

            nexti.append(i)
            nextr.append(r)
            nextvars.append(Zvar)

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

    def _fit_gp(self, x, y, name):
        self.debug("Fitting parameters for GP over %s ..." % name, level=2)

        # parameters / options
        ntry = self.opt['ntry_bq']
        verbose = self.opt['verbose'] > 3

        allkeys = ('h', 'w', 'p', 's')
        # minimization bounds
        allbounds = {
            'h': (EPS, None),
            'w': (np.radians(self.opt['bq_wmin']),
                  np.radians(self.opt['bq_wmax'])),
            'p': (EPS, None),
            's': (0, None)
        }
        # random initial value functions
        randf = {
            'h': lambda: np.random.uniform(EPS, np.max(np.abs(y))*2),
            'w': lambda: np.random.uniform(np.ptp(x) / 100., np.ptp(x) / 10.),
            'p': lambda: np.random.uniform(EPS, 2*np.pi),
            's': lambda: np.random.uniform(0, np.sqrt(np.var(y)))
        }

        # default parameter values
        default = [self.opt.get(k, None) for k in allkeys]
        # parameters we are actually fitting
        fitidx, fitkeys = zip(*[
            (i, allkeys[i]) for i in xrange(len(default))
            if default[i] is None])
        if len(fitidx) == 1:
            fitidx = [list(fitidx)]
        bounds = tuple(allbounds[key] for key in fitkeys)
        # create the GP object with dummy init params
        params = dict(zip(
            allkeys,
            [1 if default[i] is None else default[i]
             for i in xrange(len(allkeys))]))
        gp = GP(kernel(*[params[k] for k in allkeys]), x, y, self.R)

        # update the GP object with new parameter values
        def update(theta):
            params.update(dict(zip(fitkeys, theta)))
            gp.params = tuple(params[k] for k in allkeys)

        # negative log likelihood
        def f(theta):
            update(theta)
            out = -gp.log_lh()
            return out

        # jacobian of the negative log likelihood
        def df(theta):
            update(theta)
            out = -gp.dloglh_dtheta()[fitidx]
            return out

        # run the optimization a few times to find the best fit
        args = np.empty((ntry, len(bounds)))
        fval = np.empty(ntry)
        for i in xrange(ntry):
            p0 = tuple(randf[k]() for k in fitkeys)
            update(p0)
            if verbose:
                print "      p0 = %s" % (p0,)
            popt = optim.minimize(
                fun=f, x0=p0, jac=df, method='L-BFGS-B', bounds=bounds)
            args[i] = popt['x']
            fval[i] = popt['fun']
            if self.opt['verbose'] > 3:
                print "      -MLL(%s) = %f" % (args[i], fval[i])

        # choose the parameters that give the best MLL
        if args is None or fval is None:
            raise RuntimeError("Could not find MLII parameter estimates")
        update(args[np.argmin(fval)])

        self.debug("Best parameters: %s" % (gp.params,), level=2)
        return gp

    def choose_candidates(self):
        nc = len(self.ix) * 2
        ideal = list(np.linspace(0, self.R.size, nc+1).astype('i8')[:-1])
        for i in self.ix:
            diff = np.abs(np.array(ideal) - i)
            closest = np.argmin(diff)
            if diff[closest] <= self.opt['dr']:
                del ideal[closest]
        cix = sorted(ideal + self.ix)
        self.Rc = self.R[cix].copy()
        self.Dc = self.delta[cix].copy()

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
        self.log_Si = self.log_S[self.ix].copy()

        self.gp_S = self._fit_gp(self.Ri, self.Si, "S")
        self.gp_logS = self._fit_gp(self.Ri, self.log_Si, "log(S)")

        # use a crude thresholding here as our tilde transformation
        # will fail if the mean goes below zero
        self._S0 = np.clip(self.gp_S.m, EPS, np.inf)

        # fit delta, the difference between S and logS
        self.delta = self.gp_logS.m - self.log_transform(self._S0)
        self.choose_candidates()
        self.gp_Dc = self._fit_gp(self.Rc, self.Dc, "Delta_c")

        self.S_mean = self._S0 + (self._S0 + self.opt['gamma'])*self.gp_Dc.m
        # the estimated variance of S
        dm_dw, Cw = self.dm_dw, self.Cw
        self.S_cov = self.gp_logS.C + dot(dm_dw, dot(Cw, dm_dw.T))
        self.S_cov[np.abs(self.S_cov) < np.sqrt(EPS)] = EPS
        self.S_var = np.diag(self.S_cov)

    ##################################################################
    # Mean
    @property
    def mean(self):
        mean_ev = self.E(self.S_mean)
        # sanity check
        if mean_ev < 0:
            print 'mean of evidence negative'
            print 'mean of evidence: %s' % mean_ev
            mean_ev = self._S0
        return mean_ev

    ##################################################################
    # Variance
    @property
    def dm_dw(self):
        x, xs = self.R, self.Ri
        inv_Kxx = self.gp_logS.inv_Kxx
        Kxox = self.gp_logS.Kxox
        dKxox_dw = self.gp_logS.K.dK_dw(x, xs)
        dKxx_dw = self.gp_logS.K.dK_dw(xs, xs)
        inv_dKxx_dw = dot(inv_Kxx, dot(dKxx_dw, inv_Kxx))
        dm_dw = (dot(dKxox_dw, dot(inv_Kxx, self.log_Si)) -
                 dot(Kxox, dot(inv_dKxx_dw, self.log_Si)))
        return dm_dw[:, None]

    @property
    def Cw(self):
        """The variances of our posteriors over our input scale. We assume the
        covariance matrix has zero off-diagonal elements; the posterior
        is spherical.

        """
        # H_theta is the diagonal of the hessian of the likelihood of
        # the GP over the log-likelihood with respect to its log input
        # scale.
        H_theta = self.gp_logS.d2lh_dtheta2()
        # XXX: fix this slicing
        Cw = np.array([[-1. / H_theta[1, 1]]])
        return Cw

    @property
    def var(self):
        m_S = self.gp_S.m + self.opt['gamma']
        mCm = self.S_cov * m_S[:, None] * m_S[None, :]
        var_ev = self.E(self.E(mCm))
        # sanity check
        if var_ev < 0:
            print 'variance of evidence: %s' % var_ev
            raise RuntimeError('variance of evidence negative')

        # compute coarse lower bound for variance, because we know it
        # can't go below zero
        lower = self.S_mean - np.sqrt(self.S_var)
        p_R = (lower >= EPS).astype('f8')
        p_R = p_R / (np.sum(p_R) * (self.R[1] - self.R[0]))
        var_ev_low = self.E(self.E(mCm, p_R=p_R), p_R=p_R)
        return var_ev_low, var_ev

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

        self.Z_mean = self.mean
        self.Z_var = self.var
        self.print_Z(level=0)


if __name__ == "__main__":
    import util
    import sys

    # load options
    opt = util.load_opt()

    # run each stim
    util.run_all(sys.argv[1:], BayesianQuadratureModel, opt)
