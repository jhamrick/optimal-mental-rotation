import numpy as np
import scipy.optimize as optim
import scipy.stats
from numpy import dot
from numpy.linalg import inv

from snippets.safemath import EPS
from gaussian_process import GP
import kernels


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

        if opt['kernel'] == 'gaussian':
            self.kernel = kernels.GaussianKernel
        elif opt['kernel'] == 'periodic':
            self.kernel = kernels.PeriodicKernel
        else:
            raise ValueError("invalid kernel type: %s" % opt['kernel'])

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def log_transform(self, x):
        return np.log((x / self.opt['gamma']) + 1)

    def E(self, f, axis=-1, p_R=None):
        if p_R is None:
            mu, var = self.opt['prior_R']
            p_R = scipy.stats.norm.pdf(self.R, mu, np.sqrt(var))
        pfx = f * p_R
        m = np.trapz(pfx, self.R, axis=axis)
        return m

    def _small_ups_vec(self, x, gp):
        mu, var = self.opt['prior_R']
        std = np.sqrt(var + (gp.K.w ** 2))
        vec = (gp.K.h ** 2) * scipy.stats.norm.pdf(x, mu, std)
        return vec

    def _big_ups_mat(self, x1, x2, gp1, gp2):
        mu, var = self.opt['prior_R']

        var1 = gp1.K.w ** 2
        var2 = gp2.K.w ** 2
        h = (gp1.K.h * gp2.K.h) ** 2

        x1mu = (x1[:, None] - mu) ** 2
        x2mu = (x2[None, :] - mu) ** 2
        x1x2 = (x1[:, None] - x2[None, :]) ** 2
        num = (x1mu*var2) + (x2mu*var1) + (x1x2*var)
        det = (var1*var2) + (var1*var) + (var2*var)
        mat = h * np.exp(-0.5 * num / det) / (2*np.pi*np.sqrt(det))

        return mat

    def _fit_gp(self, x, y, name, **kwargs):
        self.debug("Fitting parameters for GP over %s ..." % name, level=2)

        # parameters / options
        ntry = self.opt['ntry_bq']
        verbose = self.opt['verbose'] > 3

        if self.opt['kernel'] == "gaussian":
            allkeys = ('h', 'w', 's')
        else:
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
        default_dict = dict([(k, self.opt.get(k, None)) for k in allkeys])
        default_dict.update(kwargs)
        default = [default_dict[k] for k in allkeys]
        # parameters we are actually fitting
        fitidx, fitkeys = zip(*[
            (i, allkeys[i]) for i in xrange(len(default))
            if default[i] is None])
        fitidx = list(fitidx)
        if len(fitidx) == 1:
            fitidx = [fitidx]
        bounds = tuple(allbounds[key] for key in fitkeys)
        # create the GP object with dummy init params
        params = dict(zip(
            allkeys,
            [1 if default[i] is None else default[i]
             for i in xrange(len(allkeys))]))
        kparams = tuple(params[k] for k in allkeys[:-1])
        s = params.get('s', 0)
        gp = GP(self.kernel(*kparams), x, y, self.R, s=s)

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
        nc = max(len(self.ix) * 2, self.opt['n_candidate'])
        dist = np.degrees(self.gp_S.K.w)
        ideal = list(np.linspace(0, self.R.size, nc+1).astype('i8')[:-1])
        for i in self.ix:
            diff = np.abs(np.array(ideal) - i)
            closest = np.argmin(diff)
            if diff[closest] <= dist:
                del ideal[closest]
        cix = sorted(set(ideal + self.ix))
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
        if not self.opt.get('h', None):
            logh = np.log(self.opt['h'] + 1)
        else:
            logh = None
        self.gp_logS = self._fit_gp(self.Ri, self.log_Si, "log(S)", h=logh)

        # use a crude thresholding here as our tilde transformation
        # will fail if the mean goes below zero
        self._S0 = np.clip(self.gp_S.m, EPS, np.inf)

        # fit delta, the difference between S and logS
        self.delta = self.gp_logS.m - self.log_transform(self._S0)
        self.choose_candidates()
        self.gp_Dc = self._fit_gp(self.Rc, self.Dc, "Delta_c", h=None)

        self.S_mean = self._S0 + (self._S0 + self.opt['gamma'])*self.gp_Dc.m
        # the estimated variance of S
        dm_dw, Cw = self.dm_dw, self.Cw
        self.S_cov = self.gp_logS.C + dot(dm_dw, dot(Cw, dm_dw.T))
        self.S_cov[np.abs(self.S_cov) < np.sqrt(EPS)] = EPS

        return self.S_mean, self.S_cov

    ##################################################################
    # Mean
    @property
    def mean2(self):
        mean_ev = self.E(self.S_mean)
        # sanity check
        if mean_ev < 0:
            print 'mean of evidence negative'
            print 'mean of evidence: %s' % mean_ev
            mean_ev = self._S0
        return mean_ev

    @property
    def mean(self):
        gamma = self.opt['gamma']

        xs = self.Ri
        x_sc = self.Rc
        l_s = self.Si
        delta_tl_sc = self.Dc

        # K_l = self.gp_S.Kxx
        inv_K_l = self.gp_S.inv_Kxx

        # K_del = self.gp_Dc.Kxx
        inv_K_del = self.gp_Dc.inv_Kxx

        # calculate ups for the likelihood, where ups is defined as
        # ups_s = int K(x, x_s) p(x) dx
        ups_l = self._small_ups_vec(xs, self.gp_S)

        # calculate ups for the likelihood, where ups is defined as
        # ups_s = int K(x, x_s)  p(x) dx
        #ups_tl = self._small_ups_vec(xs, self.gp_logS)

        # compute mean of int l(x) p(x) dx given l_s
        ups_inv_K_l = dot(inv_K_l, ups_l)
        minty_l = dot(ups_inv_K_l, l_s)

        # calculate Ups for delta & the likelihood, where Ups is defined as
        # Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
        Ups_del_l = self._big_ups_mat(
            x_sc, xs, self.gp_Dc, self.gp_S)

        # compute mean of int delta(x) l(x) p(x) dx given l_s and delta_tl_sc
        del_inv_K_del = dot(inv_K_del, delta_tl_sc).T
        Ups_inv_K_del_l = dot(inv_K_l, Ups_del_l.T).T
        minty_del_l = dot(del_inv_K_del, dot(Ups_inv_K_del_l, l_s))

        # calculate ups for delta, where ups is defined as
        # ups_s = int K(x, x_s)  p(x) dx
        ups_del = self._small_ups_vec(delta_tl_sc, self.gp_Dc)
        #ups_del = self.E(self.gp_Dc.Kxxo)
        print "ups_del:"
        print ups_del
        print self.E(self.gp_Dc.Kxxo)

        # compute mean of int delta(x) p(x) dx given l_s and delta_tl_sc
        ups_inv_K_del = dot(inv_K_del, ups_del).T
        minty_del = dot(ups_inv_K_del, delta_tl_sc)
        print "minty_del:"
        print minty_del
        print self.E(self.gp_Dc.m * self.gp_S.m)

        # the correction factor due to l being non-negative
        mean_ev_correction = minty_del_l + gamma * minty_del

        # the mean evidence
        mean_ev = minty_l + mean_ev_correction
        # sanity check
        if mean_ev < 0:
            print "mean of evidence negative"
            print "mean of evidence: %s" % mean_ev
            mean_ev = minty_l

        return mean_ev

    ##################################################################
    # Variance
    @property
    def dm_dw(self):
        """Compute the partial derivative of a GP mean with respect to
        w, the input scale parameter.

        The analytic form is:
        $\frac{\partial K(x_*, x)}{\partial w}K_y^{-1}\mathbf{y} - K(x_*, x)K_y^{-1}\frac{\partial K(x, x)}{\partial w}K_y^{-1}\mathbf{y}$

        Where $K_y=K(x, x) + s^2I$

        """

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
