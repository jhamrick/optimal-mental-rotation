import numpy as np

import matplotlib.pyplot as plt
import snippets.graphing as sg
import logging

from .. import config
from ..extra import BQ
from .base import BaseModel

logger = logging.getLogger("mental_rotation.model.bq")

DTYPE = np.dtype(config.get("global", "dtype"))
EPS = np.finfo(DTYPE).eps


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
        self.bq_opts = {
            'gamma': config.getfloat("bq", "gamma"),
            'h': None,
            'w': None,
            's': config.getfloat("bq", "s"),
            'ntry': config.getint("bq", "ntry"),
            'n_candidate': config.getint("bq", "n_candidate"),
            'R_mean': config.getfloat("model", "R_mu"),
            'R_var': 1. / config.getfloat("model", "R_kappa")
        }

        if 'bq_opts' in kwargs:
            self.bq_opts.update(kwargs['bq_opts'])
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
