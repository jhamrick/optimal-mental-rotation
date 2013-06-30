import numpy as np
import scipy.optimize as optim
import scipy.stats
from numpy import dot, sum, exp
from numpy.linalg import inv
from numpy.random import uniform
from collections import OrderedDict
from itertools import izip

from snippets.safemath import EPS
from gp import GP
import kernels


def mdot(*args):
    return reduce(np.dot, args)


def mvn_logpdf(x, m, C, Z=None):
    Ci = inv(C)
    if Z is None:
        Z = -0.5 * np.linalg.slogdet(C)[1]
    d = x.shape[-1]
    const = np.log(2*np.pi)*(-d/2.) + Z
    diff = x - m
    pdf = const - 0.5 * sum(
        sum(diff[..., None, :] * Ci * diff[..., None], axis=-1), axis=-1)
    return pdf


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

        # default kernel parameter values
        if self.opt['kernel'] == "gaussian":
            params = ('h', 'w', 's')
        else:
            params = ('h', 'w', 'p', 's')
        self.default_params = OrderedDict(
            [(p, self.opt.get(p, None)) for p in params])

        # minimization bounds
        self.bounds = OrderedDict([
            ('h', (EPS, None)),
            ('w', (np.radians(self.opt['bq_wmin']),
                   np.radians(self.opt['bq_wmax']))),
            ('p', (EPS, None)),
            ('s', (0, None))
        ])

        # random initial value functions
        self.randf = OrderedDict([
            ('h', lambda x, y: uniform(EPS, np.max(np.abs(y))*2)),
            ('w', lambda x, y: uniform(np.ptp(x)/100., np.ptp(x)/10.)),
            ('p', lambda x, y: uniform(EPS, 2*np.pi)),
            ('s', lambda x, y: uniform(0, np.sqrt(np.var(y))))
        ])

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

    def _gaussint1(self, x, gp):
        n, d = x.shape
        mu, cov = self.opt['prior_R']
        h = gp.K.h ** 2
        W = (np.array(gp.K.w) * np.eye(d)) ** 2
        vec = h * exp(mvn_logpdf(x, mu, cov + W))
        return vec

    def _gaussint2(self, x1, x2, gp1, gp2):
        n1, d = x1.shape
        n2, d = x2.shape
        mu, cov = self.opt['prior_R']

        ha = gp1.K.h ** 2
        hb = gp2.K.h ** 2
        Wa = (np.array(gp1.K.w) * np.eye(d)) ** 2
        Wb = (np.array(gp2.K.w) * np.eye(d)) ** 2

        i1, i2 = np.meshgrid(np.arange(n1), np.arange(n2))
        x = np.concatenate([x1[i1.T], x2[i2.T]], axis=2)
        m = np.concatenate([mu, mu])
        C = np.concatenate([
            np.concatenate([Wa + cov, cov], axis=1),
            np.concatenate([cov, Wb + cov], axis=1)
        ], axis=0)

        mat = ha * hb * exp(mvn_logpdf(x, m, C))
        return mat

    def _gaussint3(self, x, gp1, gp2):
        n, d = x.shape
        mu, cov = self.opt['prior_R']

        ha = gp1.K.h ** 2
        hb = gp2.K.h ** 2
        Wa = (np.array(gp1.K.w) * np.eye(d)) ** 2
        Wb = (np.array(gp2.K.w) * np.eye(d)) ** 2

        G = dot(cov, inv(Wa + cov))
        Gi = inv(G)
        C = Wb + 2*cov - 2*dot(G, cov)

        N1 = exp(mvn_logpdf(x, mu, cov + Wa))
        N2 = exp(mvn_logpdf(
            x[:, None] - x[None, :],
            np.zeros(d),
            mdot(Gi, C, Gi),
            Z=np.linalg.slogdet(C)[1] * -0.5))

        mat = ha**2 * hb * N2 * N1[:, None] * N1[None, :]
        return mat

    def _gaussint4(self, x, gp1, gp2):
        n, d = x.shape
        mu, cov = self.opt['prior_R']

        ha = gp1.K.h ** 2
        hb = gp2.K.h ** 2
        Wa = (np.array(gp1.K.w) * np.eye(d)) ** 2
        Wb = (np.array(gp2.K.w) * np.eye(d)) ** 2

        C0 = Wa + 2*cov
        C1 = Wb + cov - mdot(cov, inv(C0), cov)

        N0 = exp(mvn_logpdf(np.zeros(d), np.zeros(d), C0))
        N1 = exp(mvn_logpdf(x, mu, C1))

        vec = ha * hb * N0 * N1
        return vec

    def _gaussint5(self, d, gp):
        mu, cov = self.opt['prior_R']
        h = gp.K.h ** 2
        W = (np.array(gp.K.w) * np.eye(d)) ** 2
        const = h * exp(mvn_logpdf(
            np.zeros(d),
            np.zeros(d),
            W + 2*cov))
        return const

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

        # default parameter values
        default = self.default_params.copy()
        default.update({p: kwargs[p] for p in default if p in kwargs})

        # parameters we are actually fitting
        fitmask = np.array([v is None for k, v in default.iteritems()])
        fitkeys = [k for m, k in izip(fitmask, default) if m]
        bounds = tuple(self.bounds[key] for key in fitkeys)
        # create the GP object with dummy init params
        params = OrderedDict([
            (k, (1 if v is None else v))
            for k, v in default.iteritems()
        ])
        kparams = tuple(v for k, v in params.iteritems() if k != 's')
        s = params.get('s', 0)
        gp = GP(self.kernel(*kparams), x, y, s=s)

        # update the GP object with new parameter values
        def update(theta):
            params.update(dict(zip(fitkeys, theta)))
            gp.params = params.values()

        # negative log likelihood
        def f(theta):
            update(theta)
            out = -gp.log_lh
            return out

        # jacobian of the negative log likelihood
        def df(theta):
            update(theta)
            out = -gp.dloglh_dtheta[fitmask]
            return out

        # run the optimization a few times to find the best fit
        args = np.empty((ntry, len(bounds)))
        fval = np.empty(ntry)
        for i in xrange(ntry):
            p0 = tuple(self.randf[k](x, y) for k in fitkeys)
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

        Ri = self.Ri[:, None]
        Si = self.Si[:, None]
        log_Si = self.log_Si[:, None]

        self.gp_S = self._fit_gp(Ri, Si, "S")
        if self.opt.get('h', None):
            logh = np.log(self.opt['h'] + 1)
        else:
            logh = None
        self.gp_logS = self._fit_gp(Ri, log_Si, "log(S)", h=logh)

        # use a crude thresholding here as our tilde transformation
        # will fail if the mean goes below zero
        self._S0 = np.clip(self.gp_S.mean(self.R)[:, 0], EPS, np.inf)

        # fit delta, the difference between S and logS
        mls = self.gp_logS.mean(self.R)[:, 0]
        lms = self.log_transform(self._S0)
        self.delta = mls - lms
        self.choose_candidates()

        Rc = self.Rc[:, None]
        Dc = self.Dc[:, None]
        self.gp_Dc = self._fit_gp(Rc, Dc, "Delta_c", h=None)

        # the estimated mean of S
        m_S = self.gp_S.mean(self.R)[:, 0]
        m_Dc = self.gp_Dc.mean(self.R)[:, 0]
        self.S_mean = m_S + (m_S + self.gamma)*m_Dc

        # the estimated variance of S
        C_logS = self.gp_logS.cov(self.R)
        dm_dw, Cw = self.dm_dw, self.Cw
        self.S_cov = C_logS# + mdot(dm_dw, Cw, dm_dw.T)
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
        int_K_l = self._gaussint1(x_s, self.gp_S)
        E_m_l = dot(int_K_l, alpha_l)

        ## Second term
        # E[m_l*m_del | x_s, x_c] = alpha_del(x_sc)' *
        #     int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
        #     alpha_l(x_s)
        int_K_del_K_l = self._gaussint2(
            x_sc, x_s, self.gp_Dc, self.gp_S)
        E_m_l_m_del = mdot(alpha_del.T,
                           int_K_del_K_l,
                           alpha_l)

        ## Third term
        # E[m_del | x_sc] = (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
        int_K_del = self._gaussint1(x_sc, self.gp_Dc)
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

        x, x_s = self.R, self.Ri
        inv_Kxx = self.gp_logS.inv_Kxx
        Kxox = self.gp_logS.Kxox(x)
        dKxox_dw = self.gp_logS.K.dK_dw(x, x_s)
        dKxx_dw = self.gp_logS.K.dK_dw(x_s, x_s)
        inv_dKxx_dw = mdot(inv_Kxx, dKxx_dw, inv_Kxx)
        dm_dw = (mdot(dKxox_dw, inv_Kxx, self.log_Si) -
                 mdot(Kxox, inv_dKxx_dw, self.log_Si))
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
        H_theta = self.gp_logS.d2lh_dtheta2
        # XXX: fix this slicing
        Cw = np.array([[-1. / H_theta[1, 1]]])
        return Cw

    @property
    def var2(self):
        m_S = self.gp_S.mean(self.R) + self.gamma
        mCm = self.S_cov * m_S[:, None] * m_S[None, :]
        var_ev = self.E(self.E(mCm))
        # sanity check
        if var_ev < 0:
            print 'variance of evidence: %s' % var_ev
            raise RuntimeError('variance of evidence negative')
        return var_ev, var_ev

    @property
    def var(self):
        # values for the GPs over l(x) and log(l(x))
        x_s = self.gp_S.x
        n, d = x_s.shape

        alpha_l = self.gp_S.inv_Kxx_y
        alpha_tl = self.gp_logS.inv_Kxx_y
        inv_L_tl = self.gp_logS.inv_Lxx
        inv_K_tl = self.gp_logS.inv_Kxx

        ## First term
        # E[m_l C_tl m_l | x_s] = alpha_l(x_s)' *
        #    int int K_l(x_s, x) K_tl(x, x') K_l(x', x_s) p(x) p(x') dx dx' *
        #    alpha_l(x_s) - beta(x_s)'beta(x_s)
        # Where beta is defined as:
        # beta(x_s) = inv(L_tl(x_s, x_s)) *
        #    int K_tl(x_s, x) K_l(x, x_s) p(x) dx *
        #    alpha_l(x_s)
        int_K_l_K_tl_K_l = self._gaussint3(
            x_s, self.gp_S, self.gp_logS)
        int_K_tl_K_l_mat = self._gaussint2(x_s, x_s, self.gp_logS, self.gp_S)
        beta = mdot(inv_L_tl, int_K_tl_K_l_mat, alpha_l)
        alpha_int_alpha = mdot(alpha_l.T, int_K_l_K_tl_K_l, alpha_l)
        beta2 = dot(beta.T, beta)
        E_m_l_C_tl_m_l = alpha_int_alpha - beta2

        ## Second term
        # E[m_l C_tl | x_s] =
        #    [ int int K_tl(x', x) K_l(x, x_s) p(x) p(x') dx dx' -
        #      ( int K_tl(x, x_s) p(x) dx) *
        #        inv(K_tl(x_s, x_s)) *
        #        int K_tl(x_s, x) K_l(x, x_s) p(x) dx
        #      )
        #    ] alpha_l(x_s)
        int_K_tl_K_l_vec = self._gaussint4(x_s, self.gp_logS, self.gp_S)
        int_K_tl_vec = self._gaussint1(x_s, self.gp_logS)
        int_inv_int = mdot(int_K_tl_vec, inv_K_tl, int_K_tl_K_l_mat)
        E_m_l_C_tl = dot(int_K_tl_K_l_vec - int_inv_int, alpha_l)

        ## Third term
        # E[C_tl | x_s] =
        #    int int K_tl(x, x') p(x) p(x') dx dx' -
        #    ( int K_tl(x, x_s) p(x) dx *
        #      inv(K_tl(x_s, x_s)) *
        #      [int K_tl(x, x_s) p(x) dx]'
        #    )
        # Where eta is defined as:
        # eta(x_s) = inv(L_tl(x_s, x_s)) int K_tl(x_s, x) p(x) dx
        int_K_tl_scalar = self._gaussint5(d, self.gp_logS)
        E_C_tl = int_K_tl_scalar - mdot(int_K_tl_vec, inv_K_tl, int_K_tl_vec.T)

        term1 = E_m_l_C_tl_m_l
        term2 = 2*self.gamma * E_m_l_C_tl
        term3 = self.gamma**2 * E_C_tl
        V_Z = term1 + term2 + term3
        print V_Z

        ##############################################################
        ## Variance correction

        dK_tl_dw = self.gp_logS.K.dK_dw(x_s, x_s)
        zeta = dot(inv_K_tl, dK_tl_dw)
        dK_const1, dK_const2 = self._dtheta_consts(x_s)

        ## First term of nu
        term1a = mdot(alpha_l.T, int_K_tl_K_l_mat * dK_const1, alpha_tl)
        term1b = mdot(alpha_l.T, int_K_tl_K_l_mat, zeta, alpha_tl)
        term1 = term1a - term1b

        ## Second term of nu
        term2a = mdot(int_K_tl_vec * dK_const2, alpha_tl)
        term2b = mdot(int_K_tl_vec, zeta, alpha_tl)
        term2 = term2a - term2b

        nu = term1 + self.gamma*term2
        V_Z_correction = -mdot(nu, self.Cw, nu.T)
        V_Z += V_Z_correction
        print V_Z

        return V_Z, V_Z

    @property
    def var3(self):
        gamma = self.gamma

        x_s = self.Ri
        l_s = self.Si
        tl_s = self.log_Si

        inv_K_l = self.gp_S.inv_Kxx
        inv_K_l_l = dot(inv_K_l, l_s)
        inv_K_tl = self.gp_logS.inv_Kxx
        inv_K_tl_tl = dot(inv_K_tl, tl_s)
        R_tl = np.linalg.cholesky(self.gp_logS.Kxx)

        # calculate ups for the likelihood, where ups is defined as
        # ups_s = int K(x, x_s)  p(x) dx
        ups_tl = self._gaussint1(x_s, self.gp_logS)

        # calculate ups2 for the likelihood, where ups2 is defined as
        # ups2_s = int int K(x, x') K(x', x_s) p(x) prior(x') dx dx'
        ups2_l = self._gaussint4(x_s, self.gp_logS, self.gp_S)

        # calculate chi for the likelihood, where chi is defined as
        # chi = int int K(x, x') p(x) prior(x') dx dx'
        chi_tl = self._gaussint5(self.gp_logS)

        # calculate Chi for the likelihood, where Chi is defined as
        # Chi_l = int int K(x_s, x) K(x, x') K(x', x_s) p(x)
        # prior(x') dx dx'
        Chi_l_tl_l = self._gaussint3(x_s, self.gp_S, self.gp_logS)

        # calculate Ups for the likelihood and the likelihood, where
        # Ups is defined as
        # Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
        Ups_tl_l = self._gaussint2(x_s, x_s, self.gp_logS, self.gp_S)

        # compute int dx p(x) int dx' p(x') m_(l|s)(x)
        #               C_(tl|s)(x, x') m_(l|s)(x')
        Ups_inv_K_tl_l = dot(Ups_tl_l, inv_K_l_l)
        inv_R_Ups_inv_K_tl_l = dot(R_tl.T, Ups_inv_K_tl_l)
        mCminty_l_tl_l = dot(
            inv_K_l_l.T, dot(
                Chi_l_tl_l, inv_K_l_l
            )) - sum(inv_R_Ups_inv_K_tl_l ** 2)

        # compute int dx p(x) int dx' p(x') C_(tl|s)(x, x') m_(l|s)(x')
        ups_inv_K_tl = dot(inv_K_tl, ups_tl).T
        Cminty_tl_l = (dot(ups2_l.T, inv_K_l_l) -
                       dot(ups_inv_K_tl, Ups_inv_K_tl_l))

        # compute the variance of int log_transform(l)(x) p(x) dx given l_s
        Vinty_tl = chi_tl - dot(ups_inv_K_tl, ups_tl)

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
        Dtheta_K_tl = self.gp_logS.K.dK_dw(x_s, x_s)

        # compute mean of int tl(x) l(x) p(x) dx given l_s and tl_s
        minty_tl_l = dot(inv_K_tl_tl.T, Ups_inv_K_tl_l)

        # compute mean of int tl(x) p(x) dx given l_s and tl_s
        minty_tl = dot(ups_tl.T, inv_K_tl_tl)

        # Dtheta_Ups_tl_l is the modification of Ups_tl_l to allow for
        # derivatives wrt log input scales: each plate in the stack is the
        # derivative with respect to a different log input scale.
        Dtheta_Ups_tl_l_const, Dtheta_ups_tl_const = self._dtheta_consts(x_s)
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

        x_s = self.Ri
        xc = self.Rc
        x_sa = np.array(list(x_s) + [x_a])
        x_sca = np.array(list(xc) + [x_a])
        l_s = self.Si
        tl_s = self.log_Si

        K_l = self.gp_S.K
        K_tl = self.gp_logS.K
        K_del = self.gp_Dc.K

        inv_K_tl = self.gp_logS.inv_Kxx
        inv_K_tl_s = dot(inv_K_tl, tl_s)
        K_tl_s_a = K_tl(x_s, np.array([x_a]))[:, 0]

        inv_R_tl_s = self.gp_logS.inv_Lxx

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
        # the transformed likelihood between x_a and x_s: each plate in the
        # stack is the derivative with respect to a different log input scale
        Dtheta_K_tl_s = K_tl.dK_dw(x_s, x_s)
        Dtheta_K_tl_a_s = K_tl.dK_dw(x_s, np.array([x_a]))[:, 0]

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
        ups_l_sa = self._gaussint1(x_sa, self.gp_S)
        ups_inv_K_l_sa = dot(inv_K_l_sa, ups_l_sa).T

        # update Ups for delta & the likelihood, where Ups is defined as
        # Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
        Ups_sca_sa = self._gaussint2(x_sca, x_sa, self.gp_Dc, self.gp_S)

        # update for the influence of the new observation at x_a on delta.

        # update ups for delta, where ups is defined as
        # ups_s = int K(x, x_s)  prior(x) dx
        ups_del_sca = self._gaussint1(x_sca, self.gp_Dc)
        ups_inv_K_del_sca = dot(inv_K_del_sca, ups_del_sca).T

        # delta will certainly be zero at x_a
        del_sca = np.concatenate([self.Dc, np.zeros(1)])

        del_inv_K = dot(inv_K_del_sca, del_sca).T
        del_inv_K_Ups_inv_K_l_sa = dot(
            del_inv_K, dot(inv_K_l_sa, Ups_sca_sa.T).T)

        # mean of int delta(x) p(x) dx given del_sca
        minty_del = dot(ups_inv_K_del_sca, del_sca)

        # Now we finish up by subtracting the expected squared mean from the
        # previously computed second moment
        # ======================================================

        # nlsa is the vector of weights, which, when multipled with the
        # likelihoods, gives us our mean estimate for the evidence
        nlsa = del_inv_K_Ups_inv_K_l_sa + ups_inv_K_l_sa
        nla = nlsa[-1]
        nls = dot(nlsa[:-1], l_s) + (gamma * minty_del)

        e = exp(tm_a + 0.5 * tv_a)
        e2 = exp(2*tm_a + 2*tv_a)

        term1 = nls ** 2
        term2 = 2 * nls * nla * gamma * (e - 1)
        term3 = (nla * gamma) ** 2 * (e2 - 2*e + 1)

        unscaled_xpc_sqd_mean = term1 + term2 + term3
        xpc_unc = -unscaled_xpc_sqd_mean
        return xpc_unc
