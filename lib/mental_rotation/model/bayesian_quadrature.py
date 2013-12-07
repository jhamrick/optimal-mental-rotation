import numpy as np

from numpy.random import uniform
from collections import OrderedDict
from itertools import izip
from gp import GP, GaussianKernel

import matplotlib.pyplot as plt
import snippets.graphing as sg

from .base import BaseModel

DTYPE = np.float64
EPS = np.finfo(DTYPE).eps


def mdot(*args):
    return reduce(np.dot, args)


def improve_covariance_conditioning(M):
    sqd_jitters = np.max([EPS, np.max(M)]) * 1e-4
    M += np.eye(M.shape[0]) * sqd_jitters


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

    def __init__(self, R, S, **opt):

        self.gamma = DTYPE(opt['gamma'])

        self.R = np.array(R, dtype=DTYPE, copy=True)
        self.S = np.array(S, dtype=DTYPE, copy=True)
        self.log_S = self.log_transform(self.S)

        # default kernel parameter values
        self.default_params = OrderedDict()
        self.default_params['h'] = opt.get('h', None)
        self.default_params['w'] = opt.get('w', None)
        self.default_params['s'] = opt.get('s', None)

        self.ntry = opt['ntry']
        self.loglevel = opt['loglevel']
        self.n_candidate = opt['n_candidate']

        # square root of smallest float value, so we can square it
        # later on
        EPS12 = np.sqrt(EPS)

        # random initial value functions
        self.randf = OrderedDict([
            ('h', lambda x, y: lambda: uniform(EPS12, np.max(np.abs(y)) * 2)),
            ('w', lambda x, y: lambda: uniform(
                np.ptp(x) / 100., np.ptp(x) / 10.)),
            ('s', lambda x, y: lambda: uniform(0, np.sqrt(np.var(y))))
        ])

    def log_transform(self, x):
        return np.log((x / self.gamma) + 1)

    def _fit_gp(self, x, y, name, **kwargs):
        print "Fitting parameters for GP over %s ..." % name

        # default parameter values
        h = kwargs.get('h', self.default_params.get('h', None))
        w = kwargs.get('w', self.default_params.get('w', None))
        s = kwargs.get('s', self.default_params.get('s', None))

        # parameters we are actually fitting
        fitmask = np.array([v is None for v in (h, w, s)])

        # create the GP object with dummy init params
        kparams = tuple([1 if v is None else v for v in (h, w)])
        gp = GP(GaussianKernel(*kparams), x, y, s=0 if s is None else s)

        gp.fit_MLII(fitmask,
                    p0=kwargs.get('p0', None),
                    nrestart=kwargs.get('ntry', self.ntry),
                    verbose=True)

        print "Best parameters: %s" % (gp.params,)
        return gp

    def choose_candidates(self):
        ns, = self.R.shape
        d = 1
        nc = self.n_candidate
        idx = np.random.randint(0, ns, nc)
        direction = (np.random.randint(0, 2, (nc, d)) * 2) - 1
        w = self.gp_S.params[1]
        Rc = (self.R[idx] + (direction * w)) % 2 * np.pi
        Rc_no_s = np.setdiff1d(
            np.round(Rc, decimals=8),
            np.round(self.R, decimals=8))
        Rsc = np.concatenate([self.R, Rc_no_s], axis=0)
        return Rsc

    def compute_delta(self, R):
        # use a crude thresholding here as our tilde transformation
        # will fail if the mean goes below zero
        m_S = np.clip(self.gp_S.mean(R), EPS, np.inf)
        mls = self.gp_log_S.mean(R)
        lms = self.log_transform(m_S)
        delta = mls - lms
        return delta

    def fit(self):
        """Run the GP regressions to fit the likelihood function.

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        print "Fitting likelihood"

        # first figure out some sane parameters for h and w
        self.gp_S = self._fit_gp(self.R, self.S, "S")
        # then refit just w using the h we found
        self.gp_S = self._fit_gp(
            self.R, self.S, "S", h=self.gp_S.params[0], ntry=1)
        K_l = self.gp_S.Kxx
        improve_covariance_conditioning(K_l)
        assert (K_l == self.gp_S.Kxx).all()

        # use h based on the one we found for S
        logh = np.log(self.gp_S.params[0] + 1)
        self.gp_log_S = self._fit_gp(
            self.R, self.log_S, "log(S)", h=logh, ntry=1,
            p0=(self.gp_S.params[1],))
        K_tl = self.gp_log_S.Kxx
        improve_covariance_conditioning(K_tl)
        assert (K_tl == self.gp_log_S.Kxx).all()

        # fit delta, the difference between S and log_S
        self.Rc = self.choose_candidates()
        self.Dc = self.compute_delta(self.Rc)

        self.gp_Dc = self._fit_gp(self.Rc, self.Dc, "Delta_c", h=None, s=0)
        K_del = self.gp_Dc.Kxx
        improve_covariance_conditioning(K_del)
        assert (K_del == self.gp_Dc.Kxx).all()

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


class BayesianQuadratureModel(BaseModel):
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
        self.bq_opts = kwargs['bq_opts']
        del kwargs['bq_opts']

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)

    def sample(self, verbose=0):
        super(BaseModel, self).sample(iter=7, verbose=verbose)

        Ri = self.R_i
        Si = self.S_i
        ix = np.argsort(Ri)

        self.bq = BQ(Ri[ix], Si[ix], **self.bq_opts)
        self.bq.fit()

    def draw(self):
        self.model['R'].value = self.model['R'].value + np.radians(20)

    ##################################################################
    # The estimated S function

    def log_S(self, R):
        return np.log(self.S(R))

    def S(self, R):
        return self.bq.S_mean(R)

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    def log_dZ_dR(self, R):
        return np.log(self.dZ_dR(R))

    def dZ_dR(self, R):
        raise NotImplementedError

    @property
    def log_Z(self):
        return np.log(self.Z)

    @property
    def Z(self):
        raise NotImplementedError

    ##################################################################
    # Plotting methods

    def plot_S_gp(self, ax):
        Ri = self.R_i
        Si = self.S_i
        R = np.linspace(0, 2 * np.pi, 360)

        # plot the regression for S
        self._plot(
            ax, None, None, Ri, Si,
            R, self.bq.gp_S.mean(R), np.diag(self.bq.gp_S.cov(R)),
            title="GPR for $S$",
            xlabel=None,
            legend=False)
        sg.no_xticklabels(ax=ax)

    def plot_log_S_gp(self, ax):
        Ri = self.R_i
        log_Si = self.bq.log_transform(self.S_i)
        R = np.linspace(0, 2 * np.pi, 360)

        # plot the regression for log S
        self._plot(
            ax, None, None, Ri, log_Si,
            R, self.bq.gp_log_S.mean(R), np.diag(self.bq.gp_log_S.cov(R)),
            title=r"GPR for $\log(S+1)$",
            ylabel=r"Similarity ($\log(S+1)$)",
            legend=False)

    def plot_S(self, ax):
        Ri = self.R_i
        Si = self.S_i
        R = np.linspace(0, 2 * np.pi, 360)

        # combine the two regression means
        self._plot(
            ax, None, None, Ri, Si,
            R, self.bq.S_mean(R), np.diag(self.bq.S_cov(R)),
            title=r"Final GPR for $S$",
            xlabel=None,
            ylabel=None,
            legend=True)
        sg.no_xticklabels(ax=ax)

    def plot_Dc_gp(self, ax):
        R = np.linspace(0, 2 * np.pi, 360)
        delta = self.bq.compute_delta(R)
        Rc = self.bq.Rc
        Dc = self.bq.Dc

        # plot the regression for mu_log_S - log_muS
        self._plot(
            ax, R, delta, Rc, Dc,
            R, self.bq.gp_Dc.mean(R), np.diag(self.bq.gp_Dc.cov(R)),
            title=r"GPR for $\Delta_c$",
            ylabel=r"Difference ($\Delta_c$)",
            legend=False)
        yt, ytl = plt.yticks()

    def plot(self, axes):
        self.plot_S_gp(axes[0, 0])
        self.plot_S(axes[0, 1])
        self.plot_log_S_gp(axes[1, 0])
        self.plot_Dc_gp(axes[1, 1])

        # align y-axis labels
        sg.align_ylabels(-0.12, axes[0, 0], axes[1, 0])
        sg.set_ylabel_coords(-0.16, ax=axes[1, 1])
        # sync y-axis limits
        lim = (-0.2, 0.5)
        axes[0, 0].set_ylim(*lim)
        axes[0, 1].set_ylim(*lim)
        axes[1, 0].set_ylim(*lim)

        # overall figure settings
        sg.set_figsize(9, 5)
        plt.tight_layout()
