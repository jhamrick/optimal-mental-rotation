import numpy as np

import matplotlib.pyplot as plt
import snippets.graphing as sg
import logging
from bayesian_quadrature import BQ
from gp import PeriodicKernel

from .. import config
from . import BaseModel

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
            'n_candidate': config.getint("bq", "n_candidate"),
            'x_mean': config.getfloat("model", "R_mu"),
            'x_var': 1. / config.getfloat("model", "R_kappa"),
            'candidate_thresh': config.getfloat("model", "step"),
            'kernel': PeriodicKernel
        }

        if 'bq_opts' in kwargs:
            self.bq_opts.update(kwargs['bq_opts'])
            del kwargs['bq_opts']

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)

    def _init_bq(self):
        Ri = self._unwrap(self.R_i)

        ix = np.argsort(Ri)
        Ri = Ri[ix]
        Si = self.S_i[ix]

        self.bq = BQ(Ri, Si, **self.bq_opts)
        self.bq.fit_log_l((np.sqrt(7), np.pi / 3., 1, 0))
        self.bq.fit_l((np.sqrt(0.15), np.pi / 4., 1, 0))

    def sample(self, verbose=0):
        #niter = int(4 * np.pi / self.opts['step'])
        niter = 7
        super(BaseModel, self).sample(iter=niter, verbose=verbose)

    def _loop(self):
        if self._current_iter == 0:
            self.tally()
            self._current_iter += 1
            self._init_bq()
        super(BaseModel, self)._loop()

    def _dist_cost(self, x, B=2):
        R = float(self.model['R'].value)
        dist = 1. / np.exp(B * (R - x) ** 2)
        return dist

    def draw(self):
        target = self.bq.choose_next(n=1)
        print "[%d] target = %s" % (self._current_iter, target)

        self.model['R'].value = target
        self._init_bq()

        hyp = self.hypothesis_test()
        if hyp == 0 or hyp == 1:
            self.status = 'halt'

    ##################################################################
    # The estimated S function

    def log_S(self, R):
        return np.log(self.S(R))

    def S(self, R):
        try:
            len(R)
        except TypeError:
            R = np.array([R], dtype=DTYPE)
        return self.bq.l_mean(R)

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    @property
    def log_Z(self):
        Z = self.Z
        out = np.empty_like(Z)
        out[Z == 0] = -np.inf
        out[Z != 0] = np.log(Z[Z != 0])
        return out

    @property
    def Z(self):
        mean = self.bq.Z_mean()
        var = self.bq.Z_var()
        lower = mean - 1.96 * np.sqrt(var)
        upper = mean + 1.96 * np.sqrt(var)
        out = np.array([mean, lower, upper])
        out[out < 0] = 0
        return out

    ##################################################################
    # Plotting methods

    xmin = -np.pi
    xmax = np.pi

    def plot_log_S_gp(self, ax, f_S=None):
        # plot the regression for S
        self.bq.plot_gp_log_l(ax, f_l=f_S, xmin=self.xmin, xmax=self.xmax)
        ax.set_title(r"GPR for $\log(S)$")
        ax.set_ylabel(r"Similarity ($\log(S)$)")

    def plot_S_gp(self, ax, f_S=None):
        # plot the regression for S
        self.bq.plot_gp_l(ax, f_l=f_S, xmin=self.xmin, xmax=self.xmax)
        ax.set_title(r"GPR for $S$")
        ax.set_xlabel("")
        ax.set_ylabel(r"Similarity ($S$)")

    def plot_S(self, ax, f_S=None):
        # plot the regression for S
        self.bq.plot_l(ax, f_l=f_S, xmin=self.xmin, xmax=self.xmax)
        ax.set_title(r"Final GPR for $S$")
        ax.set_xlabel("")
        ax.set_ylabel("")

    def plot(self, axes, f_S=None):
        self.plot_log_S_gp(axes[0], f_S=f_S)
        self.plot_S_gp(axes[1], f_S=f_S)
        self.plot_S(axes[2], f_S=f_S)

        # align y-axis labels
        sg.align_ylabels(-0.12, axes[0], axes[1])

        # sync y-axis limits
        lim = (-0.05, 0.2)
        axes[1].set_ylim(*lim)
        axes[2].set_ylim(*lim)

        # overall figure settings
        sg.set_figsize(12, 4)
        plt.tight_layout()

    ##################################################################
    # Misc

    def hypothesis_test(self):
        llh0 = self.log_lh_h0
        llh1 = self.log_lh_h1
        if llh0 < llh1[1]: # pragma: no cover
            return 1
        elif llh0 > llh1[2]: # pragma: no cover
            return 0
        else:
            return None

    def print_stats(self):
        Z, lower, upper = self.Z
        llh0 = self.log_lh_h0
        llh1 = self.log_lh_h1
        print "Z = %f [%f, %f]" % (Z, lower, upper)
        print "log LH(h0) = %f" % llh0
        print "log LH(h1) = %f [%f, %f]" % tuple(llh1)

        llr = self.log_lh_h0 - self.log_lh_h1
        print "LH(h0) / LH(h1) = %f [%f, %f]" % tuple(llr)

        hyp = self.hypothesis_test()
        if hyp == 1: # pragma: no cover
            print "--> ACCEPT hypothesis 1"
        elif hyp == 0: # pragma: no cover
            print "--> REJECT hypothesis 1"
        else: # pragma: no cover
            print "--> UNDECIDED"
