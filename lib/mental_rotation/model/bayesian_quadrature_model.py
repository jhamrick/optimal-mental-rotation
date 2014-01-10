import numpy as np

import matplotlib.pyplot as plt
import snippets.graphing as sg
import logging
import scipy.stats
from bayesian_quadrature import BQ
from gp import PeriodicKernel

from .. import config
from . import BaseModel

logger = logging.getLogger("mental_rotation.model.bq")

DTYPE = np.dtype(config.get("global", "dtype"))
EPS = np.finfo(DTYPE).eps
MIN = np.log(np.exp2(np.float64(np.finfo(np.float64).minexp + 4)))


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

        self.bqs = {}
        self.direction = 0

        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)

    def sample(self, verbose=0):
        niter = 4 * np.pi / self.opts['step']
        super(BaseModel, self).sample(iter=niter, verbose=verbose)

    def _loop(self):
        super(BaseModel, self)._loop()

    def _check_halt(self):
        hyp = self.hypothesis_test()
        if hyp == 0 or hyp == 1:
            self.status = 'halt'
            return True
        return False

    def _add_observation(self, R, F):
        diff = self._unwrap(R - self.model['R'].value)
        step = self.opts['step']
        if float(self.model['F'].value) == F and np.isclose(np.abs(diff), step):
            self.direction = np.sign(diff)
        else:
            self.direction = np.sign(R)
        
        self.model['R'].value = R
        self.model['F'].value = F
        logger.debug(
            "R = %s, F = %s",
            self.model['R'].value, 
            float(self.model['F'].value))

        S = np.exp(self.model['log_S'].logp)
        bq = self.bqs[F]
        if not np.isclose(R, bq.x_s).any():
            bq.add_observation(R, S)
            if bq.gp_log_l.log_lh + bq.gp_l.log_lh < MIN:
                bq.fit_hypers(['h', 'w'])

    def _eval_actions(self):
        R = self.model['R'].value
        F = float(self.model['F'].value)
        step = self.opts['step']

        actions = np.array(sorted(set([
            (F, R + step), # rotate right
            (F, R - step), # rotate left 
            (F, step),     # go to the beginning and rotate right
            (F, -step),    # go to the beginning and rotate left
            (1 - F, step), # go to the beginning, flip, and rotate right
            (1 - F, -step) # go to the beginning, flip, and rotate right
        ])), dtype=float)

        actions[:, 1] = self._unwrap(actions[:, 1])
        loss = np.empty(actions.shape[0])

        logger.debug("Computing loss for F=0")
        params = ['h', 'w']
        F0 = actions[:, 0] == 0
        x_a = actions[F0, 1]
        fun = lambda: -self.bqs[0].expected_squared_mean(x_a)
        l0 = self.bqs[0].marginalize([fun], 100, params=params)
        loss[F0] = l0[0].mean(axis=0)

        logger.debug("Computing loss for F=1")
        params = ['h', 'w']
        F1 = actions[:, 0] == 1
        x_a = actions[F1, 1]
        fun = lambda: -self.bqs[1].expected_squared_mean(x_a)
        l1 = self.bqs[1].marginalize([fun], 100, params=params)
        loss[F1] = l1[0].mean(axis=0)

        return actions, loss

    def _init_bq(self, F):
        Ri = np.array([self.model['R'].value])
        Si = np.array([np.exp(self.model['log_S'].logp)])

        logger.debug("Initializing BQ object for F=%d", F)
        self.bqs[F] = BQ(Ri, Si, **self.bq_opts)
        self.bqs[F].init(
            params_tl=(8, np.pi / 2., 1, 0.0),
            params_l=(0.2, np.pi / 4., 1, 0.0))

    def draw(self):
        if self._current_iter == 0:
            self.model['R'].value = 0
            self.model['F'].value = 0
            self._init_bq(0)
            return

        if self._current_iter == 1:
            self.model['R'].value = 0
            self.model['F'].value = 1
            self._init_bq(1)
            return

        curr_R = self.model['R'].value
        curr_F = float(self.model['F'].value)
        
        actions, losses = self._eval_actions()
        minloss = np.min(losses)
        choices = np.nonzero(np.isclose(losses, minloss))[0]

        logger.debug("actions: \n%s", actions)
        logger.debug("losses: \n%s", losses)

        if self.direction != 0:
            next_R = self._unwrap(curr_R + (self.direction * self.opts['step']))
            logger.debug("next: %s", (curr_F, next_R))

            for i in choices:
                if tuple(actions[i]) == (curr_F, next_R):
                    best = i
                    break
            else:
                best = np.random.choice(choices)
        else:
            best = np.random.choice(choices)

        self._add_observation(actions[best, 1], actions[best, 0])
        self._check_halt()
            

    ##################################################################
    # The estimated S function

    def log_S(self, R, F):
        return np.log(self.S(R, F))

    def S(self, R, F):
        try:
            len(R)
        except TypeError:
            R = np.array([R], dtype=DTYPE)
        return self.bqs[F].l_mean(R)

    ##################################################################
    # Estimated dZ_dR and full estimate of Z

    def log_Z(self, F):
        Z = self.Z(F)
        out = np.empty_like(Z)
        out[Z == 0] = -np.inf
        out[Z != 0] = np.log(Z[Z != 0])
        return out

    def Z(self, F):
        mean = self.bqs[F].Z_mean()
        var = self.bqs[F].Z_var()
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
        m0 = self.bqs[0].Z_mean()
        V0 = self.bqs[0].Z_var()
        while V0 < 0:
            self.bqs[0].fit_hypers(['h', 'w'])
            m0 = self.bqs[0].Z_mean()
            V0 = self.bqs[0].Z_var()

        m1 = self.bqs[1].Z_mean()
        V1 = self.bqs[1].Z_var()
        while V1 < 0:
            self.bqs[1].fit_hypers(['h', 'w'])
            m1 = self.bqs[1].Z_mean()
            V1 = self.bqs[1].Z_var()

        N0 = scipy.stats.norm(m0, np.sqrt(V0))
        N1 = scipy.stats.norm(m1, np.sqrt(V1))

        test = N0.rvs(10000) > N1.rvs(10000)
        m = np.mean(test)
        s = scipy.stats.sem(test)

        if (m - s) > 0.95:
            return 0
        elif (m + s) < 0.05:
            return 1
        else:
            return None

    def print_stats(self):
        llh0 = self.log_lh_h0
        llh1 = self.log_lh_h1
        print "log LH(h0) = %f [%f, %f]" % tuple(llh0)
        print "log LH(h1) = %f [%f, %f]" % tuple(llh1)

        llr = self.log_lh_h0[0] - self.log_lh_h1[0]
        print "LH(h0) / LH(h1) = %f" % llr

        hyp = self.hypothesis_test()
        if hyp == 1: # pragma: no cover
            print "--> STOP and accept hypothesis 1 (flipped)"
        elif hyp == 0: # pragma: no cover
            print "--> STOP and accept hypothesis 0 (same)"
        else: # pragma: no cover
            print "--> UNDECIDED"
