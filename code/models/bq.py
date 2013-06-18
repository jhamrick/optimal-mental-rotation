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
        self.gamma = self.opt['gamma']

        if opt['kernel'] == 'gaussian':
            self.kernel = kernels.GaussianKernel
        elif opt['kernel'] == 'periodic':
            self.kernel = kernels.PeriodicKernel
        else:
            raise ValueError("invalid kernel type: %s" % opt['kernel'])

        self.R = R
        self.Ri = R[ix]
        self.Si = S[ix]
        self.log_Si = self.log_transform(self.Si)
        self.ix = ix

    def debug(self, msg, level=0):
        if self.opt['verbose'] > level:
            print ("  "*level) + msg

    def log_transform(self, x):
        return np.log((x / self.gamma) + 1)

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

    def _small_ups2_vec(self, x2, gp1, gp2):
        mu, var = self.opt['prior_R']

        ha = gp1.K.h ** 2
        wa = gp1.K.w ** 2
        hb = gp2.K.h ** 2
        wb = gp2.K.w ** 2

        N0 = ha / np.sqrt(2*np.pi*(wa + 2*var))
        Nv = wb + var - (var ** 2 / (wa + 2*var))
        N1c = hb / np.sqrt(2*np.pi*Nv)
        N1e = np.exp(-(x2 - mu) ** 2 / (2 * Nv))

        return N0 * N1c * N1e

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

    def _small_chi_const(self, gp):
        mu, var = self.opt['prior_R']
        h2 = gp.K.h ** 2
        w2 = gp.K.w ** 2
        return h2 / np.sqrt(2*np.pi*(w2 + 2*var))

    def _big_chi_mat(self, x, gp1, gp2):
        def gaussian(x, mu, var):
            C = 1. / np.sqrt(2*np.pi*var)
            e = np.exp(-(x - mu)**2 / (2*var))
            return C * e

        mu, var = self.opt['prior_R']
        wa = gp1.K.w ** 2
        hb = gp2.K.h ** 2
        wb = gp2.K.w ** 2

        det_input_scale = np.sqrt(
            (2*var + wb - 2*var**2/(var + wa)))
        input_scale = (
            ((var + wa)**2 * (2*var + wb - 2*var**2/(var + wa))) / var**2)

        vec_A = self._small_ups_vec(x, gp1)
        C = (hb / det_input_scale) / np.sqrt(2*np.pi*input_scale)
        e = np.exp(-(x[:, None] - x[None, :])**2 / (2*input_scale))
        mat_A = C * e

        mat = mat_A * dot(vec_A[:, None], vec_A[None])
        return mat

    def _dtheta_consts(self, x):
        mu, L = self.opt['prior_R']
        Wl = float(self.gp_S.K.w ** 2)
        Wtl = float(self.gp_logS.K.w ** 2)

        denom = (Wtl * L) + (Wl * L) + (Wl * Wtl)
        C = Wtl * L / denom
        Ct = Wl * L / denom

        xsubmu = (x - mu)[None, :]
        Dtheta_Ups_tl_l_const = (1. / Wtl) * (
            L * (1 - C - Ct) +
            (-(C * xsubmu) + ((1 - Ct) * xsubmu.T) ** 2))

        LLWtl = 1 - (L / (L + Wtl))
        Dtheta_ups_tl_const = (1. / Wtl) * (
            (L * LLWtl) + (LLWtl * xsubmu[0]) ** 2)

        return Dtheta_Ups_tl_l_const, Dtheta_ups_tl_const

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
        if self.opt.get('h', None):
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

        # the estimated mean of S
        m_S = self.gp_S.m
        m_Dc = self.gp_Dc.m
        self.S_mean = m_S + (m_S + self.gamma)*m_Dc

        # the estimated variance of S
        C_logS = self.gp_logS.C
        dm_dw, Cw = self.dm_dw, self.Cw
        self.S_cov = C_logS + dot(dm_dw, dot(Cw, dm_dw.T))
        self.S_cov[np.abs(self.S_cov) < np.sqrt(EPS)] = EPS

        return self.S_mean, self.S_cov

    ##################################################################
    # Mean
    @property
    def mean_approx(self):
        m_Z = self.E(self.S_mean)
        return m_Z

    @property
    def mean(self):
        # values for the GP over l(x)
        x_s = self.gp_S.x
        alpha_l = self.gp_S.inv_Kxx_y

        # values for the GP of Delta(x)
        x_sc = self.gp_Dc.x
        alpha_del = self.gp_Dc.inv_Kxx_y

        ## First term
        # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
        int_K_l = self._small_ups_vec(x_s, self.gp_S)
        E_m_l = dot(int_K_l, alpha_l)

        ## Second term
        # E[m_l*m_del | x_s, x_c] = alpha_del(x_sc)' *
        #     int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
        #     alpha_l(x_s)
        int_K_del_K_l = self._big_ups_mat(
            x_sc, x_s, self.gp_Dc, self.gp_S)
        E_m_l_m_del = dot(
            alpha_del.T, dot(int_K_del_K_l, alpha_l))

        ## Third term
        # E[m_del | x_sc] = (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
        int_K_del = self._small_ups_vec(x_sc, self.gp_Dc)
        E_m_del = dot(int_K_del, alpha_del)

        # put the three terms together
        m_Z = E_m_l + E_m_l_m_del + self.gamma*E_m_del
        return m_Z

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
    def var2(self):
        m_S = self.gp_S.m + self.gamma
        mCm = self.S_cov * m_S[:, None] * m_S[None, :]
        var_ev = self.E(self.E(mCm))
        # sanity check
        if var_ev < 0:
            print 'variance of evidence: %s' % var_ev
            raise RuntimeError('variance of evidence negative')

        # # compute coarse lower bound for variance, because we know it
        # # can't go below zero
        # lower = self.S_mean - np.sqrt(np.diag(self.S_cov))
        # p_R = (lower >= EPS).astype('f8')
        # p_R = p_R / (np.sum(p_R) * (self.R[1] - self.R[0]))
        # var_ev_low = self.E(self.E(mCm, p_R=p_R), p_R=p_R)
        # return var_ev_low, var_ev
        return var_ev, var_ev

    @property
    def var(self):
        gamma = self.gamma

        xs = self.Ri
        l_s = self.Si
        tl_s = self.log_Si

        inv_K_l = self.gp_S.inv_Kxx
        inv_K_l_l = dot(inv_K_l, l_s)
        inv_K_tl = self.gp_logS.inv_Kxx
        inv_K_tl_tl = dot(inv_K_tl, tl_s)
        R_tl = np.linalg.cholesky(self.gp_logS.Kxx)

        # calculate ups for the likelihood, where ups is defined as
        # ups_s = int K(x, x_s)  p(x) dx
        ups_tl = self._small_ups_vec(xs, self.gp_logS)

        # calculate ups2 for the likelihood, where ups2 is defined as
        # ups2_s = int int K(x, x') K(x', x_s) p(x) prior(x') dx dx'
        ups2_l = self._small_ups2_vec(xs, self.gp_logS, self.gp_S)

        # calculate chi for the likelihood, where chi is defined as
        # chi = int int K(x, x') p(x) prior(x') dx dx'
        chi_tl = self._small_chi_const(self.gp_logS)

        # calculate Chi for the likelihood, where Chi is defined as
        # Chi_l = int int K(x_s, x) K(x, x') K(x', x_s) p(x)
        # prior(x') dx dx'
        Chi_l_tl_l = self._big_chi_mat(xs, self.gp_S, self.gp_logS)

        # calculate Ups for the likelihood and the likelihood, where
        # Ups is defined as
        # Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
        Ups_tl_l = self._big_ups_mat(xs, xs, self.gp_logS, self.gp_S)

        # compute the variance of int log_transform(l)(x) p(x) dx given l_s
        ups_inv_K_tl = dot(inv_K_tl, ups_tl).T
        Vinty_tl = chi_tl - dot(ups_inv_K_tl, ups_tl)

        # compute int dx p(x) int dx' p(x') C_(tl|s)(x, x') m_(l|s)(x')
        Ups_inv_K_tl_l = dot(Ups_tl_l, inv_K_l_l)
        Cminty_tl_l = (dot(ups2_l.T, inv_K_l_l) -
                       dot(ups_inv_K_tl, Ups_inv_K_tl_l))

        # compute int dx p(x) int dx' p(x') m_(l|s)(x)
        #               C_(tl|s)(x, x') m_(l|s)(x')
        inv_R_Ups_inv_K_tl_l = dot(R_tl.T, Ups_inv_K_tl_l)
        mCminty_l_tl_l = dot(
            inv_K_l_l.T, dot(
                Chi_l_tl_l, inv_K_l_l
            )) - sum(inv_R_Ups_inv_K_tl_l ** 2)

        # variance of the evidence
        var_ev = (gamma**2 * Vinty_tl +
                  2*gamma * Cminty_tl_l +
                  mCminty_l_tl_l)

        ## now we account for our uncertainty in the log input scales

        # the variances of our posteriors over our input scales. We assume the
        # covariance matrix has zero off-diagonal elements; the posterior is
        # spherical.
        V_theta = self.Cw[0, 0]

        # Dtheta_K_tl is the gradient of the Gaussian covariance over the
        # transformed likelihood between x_s and x_s: each plate in the stack
        # is the derivative with respect to a different log input scale
        Dtheta_K_tl = self.gp_logS.K.dK_dw(xs, xs)

        # compute mean of int tl(x) l(x) p(x) dx given l_s and tl_s
        minty_tl_l = dot(inv_K_tl_tl.T, Ups_inv_K_tl_l)

        # compute mean of int tl(x) p(x) dx given l_s and tl_s
        minty_tl = dot(ups_tl.T, inv_K_tl_tl)

        # Dtheta_Ups_tl_l is the modification of Ups_tl_l to allow for
        # derivatives wrt log input scales: each plate in the stack is the
        # derivative with respect to a different log input scale.
        Dtheta_Ups_tl_l_const, Dtheta_ups_tl_const = self._dtheta_consts(xs)
        Dtheta_Ups_tl_l = Ups_tl_l * Dtheta_Ups_tl_l_const

        # Dtheta_ups_tl is the modification of ups_tl to allow for
        # derivatives wrt log input scales: each plate in the stack is the
        # derivative with respect to a different log input scale.
        Dtheta_ups_tl = ups_tl * Dtheta_ups_tl_const

        line1 = -minty_tl_l - (gamma * minty_tl)
        line2 = Ups_inv_K_tl_l.T + (gamma * ups_tl.T)
        line3 = dot(dot(inv_K_tl, Dtheta_K_tl), inv_K_tl_tl)
        line4 = dot(inv_K_l_l.T, dot(Dtheta_Ups_tl_l.T, inv_K_tl_tl))
        line5 = gamma * dot(Dtheta_ups_tl.T, inv_K_tl_tl)

        int_ml_Dtheta_mtl = line1 + dot(line2, line3) + line4 + line5

        # Now perform the correction to our variance
        var_ev_correction = (int_ml_Dtheta_mtl ** 2) * V_theta
        var_ev += var_ev_correction

        return var_ev, var_ev

    def expected_uncertainty_evidence(self, x_a):
        gamma = self.gamma

        xs = self.Ri
        xc = self.Rc
        x_sa = np.array(list(xs) + [x_a])
        x_sca = np.array(list(xc) + [x_a])
        l_s = self.Si
        tl_s = self.log_Si

        K_l = self.gp_S.K
        K_tl = self.gp_logS.K
        K_del = self.gp_Dc.K

        inv_K_tl = self.gp_logS.inv_Kxx
        inv_K_tl_s = dot(inv_K_tl, tl_s)
        K_tl_s_a = K_tl(xs, np.array([x_a]))[:, 0]

        R_tl_s = np.linalg.cholesky(self.gp_logS.Kxx)
        inv_R_tl_s = inv(R_tl_s)

        # compute predictive mean for transformed likelihood, given
        # zero prior mean
        tm_a = dot(K_tl_s_a.T, inv_K_tl_s)

        # compute predictive variance for transformed likelihood,
        # given zero prior mean
        inv_R_K_tl_s_a = dot(inv_R_tl_s, K_tl_s_a)
        C = K_tl(np.zeros(1), np.zeros(1))[0, 0]
        tv_a = C - sum(inv_R_K_tl_s_a ** 2)
        if tv_a < 0:
            tv_a = EPS

        # we correct for the impact of learning this new hyperparameter sample,
        # r_a, on our belief about the log input scales

        # Dtheta_K_tl_a_s is the gradient of the tl Gaussian covariance over
        # the transformed likelihood between x_a and xs: each plate in the
        # stack is the derivative with respect to a different log input scale
        Dtheta_K_tl_s = K_tl.dK_dw(xs, xs)
        Dtheta_K_tl_a_s = K_tl.dK_dw(xs, np.array([x_a]))[:, 0]

        K_inv_K_tl_a_s = dot(inv_R_tl_s, inv_R_K_tl_s_a).T

        # gradient of the mean of the log-likelihood at the added point wrt the
        # log-input scales
        Dtheta_tm_a = (dot(Dtheta_K_tl_a_s.T, inv_K_tl_s) -
                       dot(K_inv_K_tl_a_s,
                           dot(Dtheta_K_tl_s, inv_K_tl_s)))

        # Now perform the correction to our predictive variance
        V_theta = self.Cw[0, 0]
        tv_a += (Dtheta_tm_a ** 2) * V_theta

        # we update the covariance matrix over the likelihood
        K_l_sa = K_l(x_sa, x_sa)
        try:
            inv_K_l_sa = inv(K_l_sa)
        except np.linalg.LinAlgError:
            return 0

        # we update the covariance matrix over delta
        K_del_sca = K_del(x_sca, x_sca)
        try:
            inv_K_del_sca = inv(K_del_sca)
        except np.linalg.LinAlgError:
            return 0

        # Now we compute the new rows and columns of Ups and ups
        # matrices required to evaluate the sqd mean evidence after
        # adding this new trial sample
        # ======================================================

        # update ups for the likelihood, where ups is defined as
        # ups_s = int K(x, x_s)  prior(x) dx
        ups_l_sa = self._small_ups_vec(x_sa, self.gp_S)
        ups_inv_K_l_sa = dot(inv_K_l_sa, ups_l_sa).T

        # update Ups for delta & the likelihood, where Ups is defined as
        # Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
        Ups_sca_sa = self._big_ups_mat(x_sca, x_sa, self.gp_Dc, self.gp_S)

        # update for the influence of the new observation at x_a on delta.

        # update ups for delta, where ups is defined as
        # ups_s = int K(x, x_s)  prior(x) dx
        ups_del_sca = self._small_ups_vec(x_sca, self.gp_Dc)
        ups_inv_K_del_sca = dot(inv_K_del_sca, ups_del_sca).T

        # delta will certainly be zero at x_a
        delta_tl_sca = np.concatenate([self.Dc, np.zeros(1)])

        del_inv_K = dot(inv_K_del_sca, delta_tl_sca).T
        del_inv_K_Ups_inv_K_l_sa = dot(
            del_inv_K, dot(inv_K_l_sa, Ups_sca_sa.T).T)

        # mean of int delta(x) p(x) dx given delta_tl_sca
        minty_del = dot(ups_inv_K_del_sca, delta_tl_sca)

        # Now we finish up by subtracting the expected squared mean from the
        # previously computed second moment
        # ======================================================

        # nlsa is the vector of weights, which, when multipled with the
        # likelihoods, gives us our mean estimate for the evidence
        nlsa = del_inv_K_Ups_inv_K_l_sa + ups_inv_K_l_sa
        nla = nlsa[-1]
        nls = dot(nlsa[:-1], l_s) + (gamma * minty_del)

        e = np.exp(tm_a + 0.5 * tv_a)
        e2 = np.exp(2*tm_a + 2*tv_a)

        term1 = nls ** 2
        term2 = 2 * nls * nla * gamma * (e - 1)
        term3 = (nla * gamma) ** 2 * (e2 - 2*e + 1)

        unscaled_xpc_sqd_mean = term1 + term2 + term3
        xpc_unc = -unscaled_xpc_sqd_mean
        return xpc_unc
