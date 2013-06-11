import numpy as np
import scipy.optimize as optim
from numpy import dot

from snippets.safemath import EPS
from periodic_kernel import PeriodicKernel as kernel
from gaussian_process import GP


class BQ(object):
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

    def __init__(self, R, S, ix, opt):
        self.opt = opt
        self.R = R
        self.Ri = R[ix]
        self.Si = S[ix]
        self.log_Si = self.log_transform(self.Si)
        self.ix = ix

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def log_transform(self, x):
        return np.log((x / self.opt['gamma']) + 1)

    def E(self, f, axis=-1, p_R=None):
        if p_R is None:
            p_R = self.opt['prior_R']
        pfx = f * p_R
        m = np.trapz(pfx, self.R, axis=axis)
        return m

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

        return self.S_mean, self.S_cov

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
        lower = self.S_mean - np.sqrt(np.diag(self.S_cov))
        p_R = (lower >= EPS).astype('f8')
        p_R = p_R / (np.sum(p_R) * (self.R[1] - self.R[0]))
        var_ev_low = self.E(self.E(mCm, p_R=p_R), p_R=p_R)
        return var_ev_low, var_ev
