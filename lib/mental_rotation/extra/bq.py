import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})

import logging

from gp import GP, GaussianKernel
from .. import config

import bq_c

logger = logging.getLogger("mental_rotation.extra.bq")

DTYPE = np.dtype(config.get("global", "dtype"))
EPS = np.finfo(DTYPE).eps


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

    def __init__(self, R, S,
                 gamma, ntry, n_candidate,
                 R_mean, R_var,
                 h=None, w=None, s=None):

        # save the given parameters
        self.gamma = float(gamma)
        self.ntry = int(ntry)
        self.n_candidate = int(n_candidate)
        self.R_mean = float(R_mean)
        self.R_cov = np.array([[R_var]], dtype=DTYPE)

        # default kernel parameter values
        self.default_params = dict(h=h, w=w, s=s)

        self.R = np.array(R, dtype=DTYPE, copy=True)
        self.S = np.array(S, dtype=DTYPE, copy=True)

        if self.R.ndim > 1:
            raise ValueError("invalid number of dimensions for R")
        if self.S.ndim > 1:
            raise ValueError("invalid number of dimensions for S")
        if self.R.shape != self.S.shape:
            raise ValueError("shape mismatch for R and S")

        self.log_S = self.log_transform(self.S)
        self.n_sample = self.R.size

    def log_transform(self, x):
        return np.log((x / self.gamma) + 1)

    def _fit_gp(self, x, y, **kwargs):
        # figure out which parameters we are fitting and how to
        # generate them
        randf = []
        fitmask = np.empty(3, dtype=bool)
        for i, p in enumerate(['h', 'w', 's']):
            # are we fitting the parameter?
            p_v = kwargs.get(p, self.default_params.get(p, None))
            if p_v is None:
                fitmask[i] = True

            # what should we use as an initial parameter?
            p0 = kwargs.get('%s0' % p, p_v)
            if p0 is None:
                randf.append(lambda: np.abs(np.random.normal()))
            else:
                # need to use keyword argument, because python does
                # not assign new values to closure variables in loop
                def f(p=p0):
                    return p
                randf.append(f)

        # generate initial parameter values
        randf = np.array(randf)
        h, w, s = [f() for f in randf]

        # number of restarts
        ntry = kwargs.get('ntry', self.ntry)

        # create the GP object
        gp = GP(GaussianKernel(h, w), x, y, s=s)

        # fit the parameters
        gp.fit_MLII(fitmask, randf=randf[fitmask], nrestart=ntry)
        return gp

    def _choose_candidates(self):
        ns = self.n_sample
        nc = self.n_candidate

        # choose anchor points
        idx = np.random.randint(0, ns, nc)

        # compute the candidate points
        eps = np.random.choice([-1, 1], (nc, 1)) * self.gp_S.params[1]
        Rc = (self.R[idx] + eps) % (2 * np.pi)

        # make sure they don't overlap with points we already have
        Rc_no_s = np.setdiff1d(
            np.round(Rc, decimals=4),
            np.round(self.R, decimals=4))

        # make the array of old points + new points
        Rsc = np.concatenate([self.R, Rc_no_s])
        return Rsc

    def compute_delta(self, R):
        # use a crude thresholding here as our tilde transformation
        # will fail if the mean goes below zero
        m_S = np.clip(self.gp_S.mean(R), EPS, np.inf)
        mls = self.gp_log_S.mean(R)
        lms = self.log_transform(m_S)
        delta = mls - lms
        return delta

    def _fit_S(self):
        # first figure out some sane parameters for h and w
        logger.info("Fitting parameters for GP over S")
        self.gp_S = self._fit_gp(self.R, self.S)

        # then refit just w using the h we found
        logger.info("Fitting w parameter for GP over S")
        self.gp_S = self._fit_gp(
            self.R, self.S,
            h=self.gp_S.params[0],
            ntry=1)

        # try to improve the kernel matrix conditioning
        bq_c.improve_covariance_conditioning(self.gp_S.Kxx)

    def _fit_log_S(self):
        # use h based on the one we found for S
        logger.info("Fitting parameters for GP over log(S)")
        self.gp_log_S = self._fit_gp(
            self.R, self.log_S,
            h=np.log(self.gp_S.params[0] + 1),
            w0=self.gp_S.params[1],
            ntry=1)

        # try to improve the kernel matrix conditioning
        bq_c.improve_covariance_conditioning(self.gp_log_S.Kxx)

    def _fit_Dc(self):
        # choose candidate locations and compute delta, the difference
        # between S and log(S)
        self.Rc = self._choose_candidates()
        self.Dc = self.compute_delta(self.Rc)

        # fit gp parameters for delta
        logger.info("Fitting parameters for GP over Delta_c")
        self.gp_Dc = self._fit_gp(
            self.Rc, self.Dc, h=None, s=0)

        # try to improve the kernel matrix conditioning
        bq_c.improve_covariance_conditioning(self.gp_Dc.Kxx)

    def fit(self):
        """Run the GP regressions to fit the likelihood function.

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        logger.info("Fitting likelihood")

        self._fit_S()
        self._fit_log_S()
        self._fit_Dc()

    def S_mean(self, R):
        # the estimated mean of S
        m_S = self.gp_S.mean(R)
        m_Dc = self.gp_Dc.mean(R)
        S_mean = np.clip(m_S, EPS, np.inf) + (m_S + self.gamma) * m_Dc
        return S_mean

    def S_cov(self, R):
        # the estimated variance of S
        C_log_S = self.gp_log_S.cov(R)
        # dm_dw, Cw = self.dm_dw(R), self.Cw(self.gp_log_S)
        S_cov = C_log_S# + mdot(dm_dw, Cw, dm_dw.T)
        S_cov[np.abs(S_cov) < np.sqrt(EPS)] = EPS
        return S_cov

    def Z_mean(self):
        x_s = self.gp_S.x
        x_sc = self.gp_Dc.x
        ns, d = x_s.shape
        nc, d = x_sc.shape

        # values for the GP over l(x)
        alpha_l = self.gp_S.inv_Kxx_y

        # values for the GP of Delta(x)
        alpha_del = self.gp_Dc.inv_Kxx_y

        ## First term
        # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
        int_K_l = self._gaussint1(x_s, self.gp_S)
        E_m_l = float(np.dot(int_K_l, alpha_l))
        assert E_m_l > 0

        ## Second term
        # E[m_l*m_del | x_s, x_c] = alpha_del(x_sc)' *
        #     int K_del(x_sc, x) K_l(x, x_s) p(x) dx *
        #     alpha_l(x_s)
        int_K_del_K_l = self._gaussint2(
            x_sc, x_s, self.gp_Dc, self.gp_S)
        E_m_l_m_del = float(np.dot(np.dot(
            alpha_del.T, int_K_del_K_l), alpha_l))

        ## Third term
        # E[m_del | x_sc] = (int K_del(x, x_sc) p(x) dx) alpha_del(x_c)
        int_K_del = self._gaussint1(x_sc, self.gp_Dc)
        E_m_del = float(np.dot(int_K_del, alpha_del))

        # put the three terms together
        m_Z = E_m_l + E_m_l_m_del + self.gamma * E_m_del

        return m_Z

    def _gaussint1(self, x, gp):
        return bq_c.gaussint1(
            x, gp.K.h, gp.K.w, self.R_mean, self.R_cov)

    def _gaussint2(self, x1, x2, gp1, gp2):
        return bq_c.gaussint2(
            x1, x2,
            gp1.K.h, gp1.K.w,
            gp2.K.h, gp2.K.w,
            self.R_mean, self.R_cov)

    def _gaussint3(self, x, gp1, gp2):
        return bq_c.gaussint3(
            x, gp1.K.h, gp1.K.w, gp2.K.h, gp2.K.w,
            self.R_mean, self.R_cov)

    def _gaussint4(self, x, gp1, gp2):
        return bq_c.gaussint4(
            x, gp1.K.h, gp1.K.w, gp2.K.h, gp2.K.w,
            self.R_mean, self.R_cov)

    def _gaussint5(self, d, gp):
        return bq_c.gaussint5(
            d, gp.K.h, gp.K.w, self.R_mean, self.R_cov)

    def _dtheta_consts(self, x):
        return bq_c.dtheta_consts(
            x, self.gp_S.K.w, self.gp_log_S.K.w,
            self.R_mean, self.R_cov)
