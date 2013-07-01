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
        vec = h * exp(mvn_logpdf(x, mu, cov + W))[None]
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
        n, d = x.shape
        mu, L = self.opt['prior_R']
        I = np.eye(d)
        Wl = (self.gp_S.K.w ** 2) * I
        Wtl = (self.gp_logS.K.w ** 2) * I
        iWtl = inv(Wtl)

        A = dot(L, inv(Wtl + L))
        B = L - dot(A, L)
        C = dot(B, inv(B + Wl))
        CA = dot(C, A)
        BCB = B - dot(C, B)

        xsubmu = x - mu
        mat_const = np.empty((n, n, d, d))
        vec_const = np.empty((n, d, d))
        c = -0.5*iWtl

        m1 = dot(A - I, xsubmu.T).T
        m2a = dot(A - CA - I, xsubmu.T).T
        m2b = dot(C, xsubmu.T).T

        for i in xrange(n):
            vec_const[i] = c * (I + dot(B - dot(m1[i], m1[i].T), iWtl))

            for j in xrange(n):
                m2 = m2a[i] + m2b[j]
                mat_const[i, j] = c * (I + dot(BCB - dot(m2, m2.T), iWtl))

        return mat_const, vec_const

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
    def dm_dw(self, x):
        """Compute the partial derivative of a GP mean with respect to
        w, the input scale parameter.

        """
        dm_dtheta = self.gp_logS.dm_dtheta(x)
        # XXX: fix this slicing
        dm_dw = dm_dtheta[1]
        return dm_dw

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
        Cw = np.diagflat(-1. / H_theta[1, 1])
        return Cw

    @property
    def var_approx(self):
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
        V_Z = sum(term1 + term2 + term3)

        ##############################################################
        ## Variance correction

        dK_tl_dw = self.gp_logS.K.dK_dw(x_s, x_s)
        zeta = dot(inv_K_tl, dK_tl_dw)
        dK_const1, dK_const2 = self._dtheta_consts(x_s)

        ## First term of nu
        term1a = sum(sum(
            alpha_l[:, None, :, None] *
            int_K_tl_K_l_mat[:, :, None, None] *
            dK_const1 *
            alpha_tl[None, :, None, :],
            axis=0), axis=0)
        term1b = mdot(alpha_l.T, int_K_tl_K_l_mat, zeta, alpha_tl)
        term1 = term1a - term1b

        ## Second term of nu
        term2a = sum(
            int_K_tl_vec.T[:, :, None] *
            dK_const2 *
            alpha_tl[:, :, None],
            axis=0)
        term2b = mdot(int_K_tl_vec, zeta, alpha_tl)
        term2 = term2a - term2b

        nu = np.diag(term1 + self.gamma*term2)
        V_Z_correction = -mdot(nu, self.Cw, nu.T)
        V_Z += V_Z_correction

        return V_Z

    def expected_uncertainty_evidence(self, x_a):
        x_s = self.Ri[:, None]
        xc = self.Rc[:, None]

        if (x_s == x_a).all(axis=1).any():
            return 0

        # compute expected transformed mean
        tm_a = self.gp_logS.mean(x_a)

        # compute expected transformed covariance
        dm_dw = self.dm_dw(x_a)
        Cw = self.Cw
        tC_a = self.gp_logS.cov(x_a) + mdot(dm_dw, Cw, dm_dw.T)

        #####

        # include new x_a
        x_sa = np.concatenate([x_s, x_a], axis=0)
        if not (x_a == xc).all(axis=1).any():
            x_sca = np.concatenate([xc, x_a], axis=0)
        else:
            x_sca = xc
        l_a = self.gp_S.mean(x_a)
        l_s = self.Si[:, None]
        l_sa = np.concatenate([l_s, l_a])
        del_sc = self.Dc[:, None]
        del_sca = np.concatenate([del_sc, np.zeros((1, 1))])

        # update gp over S
        gp_Sa = self.gp_S.copy()
        gp_Sa.x = x_sa
        gp_Sa.y = l_sa
        inv_K_l = gp_Sa.inv_Kxx

        # update gp over delta
        gp_Dca = self.gp_Dc.copy()
        gp_Dca.x = x_sca
        gp_Dca.y = del_sca
        alpha_del = gp_Dca.inv_Kxx_y

        # compute constants

        ## First term
        # int K_l(x, x_s) p(x) dx inv(K_l(x_s, x_s))
        int_K_l = self._gaussint1(x_sa, gp_Sa)
        A = dot(int_K_l, inv_K_l)

        ## Second term
        # alpha_del(x_sc)' *
        # int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
        # inv(K_l(x_s, x_s))
        int_K_del_K_l = self._gaussint2(x_sca, x_sa, gp_Dca, gp_Sa)
        B = mdot(alpha_del.T, int_K_del_K_l, inv_K_l)

        ## Third term
        # (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
        int_K_del = self._gaussint1(x_sca, gp_Dca)
        C = dot(int_K_del, alpha_del)

        # nlsa is the vector of weights, which, when multipled with
        # the likelihoods, gives us our mean estimate for the evidence
        nlsa = A + B
        assert nlsa.shape[0] == 1
        nla = nlsa[0, [-1]]
        nls = dot(nlsa[0, :-1], l_s) + self.gamma * C

        e = tm_a + 0.5*tC_a
        e2 = 2*tm_a + 2*tC_a

        nls2 = nls ** 2
        nlsnla = 2*nls*nla * self.gamma * (e - 1)
        nla2 = (nla * self.gamma) ** 2 * (e2 - 2*e + 1)

        xpc_sqd_mean = nls2 + nlsnla + nla2
        mean_second_moment = self.mean**2 + self.var
        xpc_unc = mean_second_moment - xpc_sqd_mean
        return xpc_unc
