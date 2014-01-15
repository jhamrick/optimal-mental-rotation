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

EPS = np.finfo(np.float64).eps
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

    _iter = 10

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

    def _check_done(self):
        hyp = self.hypothesis_test()
        if hyp == 0 or hyp == 1:
            self.status = 'done'
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
            "F = %s, R = %s, direction = %s", 
            float(self.model['F'].value),
            self.model['R'].value,
            self.direction)

        S = np.exp(self.model['log_S'].logp)
        bq = self.bqs[F]
        if not np.isclose(R, bq.x_s).any():
            bq.add_observation(R, S)
            bq.fit_hypers(['w'])

    def _get_actions(self):
        R = self.model['R'].value
        F = float(self.model['F'].value)
        step = self.opts['step']

        actions = [
            (F, R + step),
            (F, R - step),
            (1 - F, R),
            (F, 0),
            (1 - F, 0),
        ]

        new_actions = []
        old_actions = []
        for action in set(actions):
            F, R = action
            if np.isclose(0, self._unwrap(R - self.bqs[F].x_s)).any():
                old_actions.append((F, R))
            else:
                new_actions.append((F, R))

        if len(new_actions) > 0:
            actions = new_actions
        else:
            actions = old_actions

        actions = np.array(sorted(actions))
        actions[:, 1] = self._unwrap(actions[:, 1])
        return actions

    def _eval_actions(self, n=50):
        actions = self._get_actions()

        loss = np.empty(actions.shape[0])
        params = ['h', 'w']
        F0 = actions[:, 0] == 0
        F1 = actions[:, 0] == 1

        if self.bqs[0].x_s.size > 1:
            self.bqs[0].fit_hypers(params)

        x_a = actions[F0, 1]
        funs = [
            lambda: self.bqs[0].expected_squared_mean_and_mean(x_a),
            self.bqs[0].Z_mean,
            self.bqs[0].Z_var
        ]
        # l0, m0, V0 = self.bqs[0].marginalize(
        #     funs, n, params=params)
        l0, m0, V0 = [np.array([f()]) for f in funs]

        if self.bqs[1].x_s.size > 1:
            self.bqs[1].fit_hypers(params)

        x_a = actions[F1, 1]
        funs = [
            lambda: self.bqs[1].expected_squared_mean_and_mean(x_a),
            self.bqs[1].Z_mean,
            self.bqs[1].Z_var
        ]
        # l1, m1, V1 = self.bqs[1].marginalize(
        #     funs, n, params=params)
        l1, m1, V1 = [np.array([f()]) for f in funs]

        esm0 = l0[..., 0].T
        em0 = l0[..., 1].T
        esm1 = l1[..., 0].T
        em1 = l1[..., 1].T

        self._m0 = max(0, m0.mean())
        self._V0 = max(0, V0.mean())
        self._m1 = max(0, m1.mean())
        self._V1 = max(0, V1.mean())

        # E_V0 = V0 + m0 - esm0
        E_V0 = esm0 - em0 ** 2
        loss[F0] = E_V0.mean(axis=1) + self._V1

        # E_V1 = V1 + m1 - esm1
        E_V1 = esm1 - em1 ** 2
        loss[F1] = self._V0 + E_V1.mean(axis=1)

        return actions, loss

    def _init_bq(self, F):
        Ri = np.array([self.model['R'].value])
        Si = np.array([np.exp(self.model['log_S'].logp)])
        
        logger.debug("Initializing BQ object for F=%d", F)
        self.bqs[F] = BQ(Ri, Si, **self.bq_opts)
        self.bqs[F].init(
            params_tl=(8, np.pi / 2., 1.0, 0.0),
            params_l=(0.2, np.pi / 4., 1.0, 0.0))

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
        if self._check_done():
            return

        minloss = np.min(losses)
        choices = np.nonzero(np.isclose(losses, minloss))[0]

        for action, loss in zip(actions, losses):
            logger.debug("L(%s) = %s" % (action, loss))

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
            

    ##################################################################
    # The estimated S function

    def log_S(self, R, F):
        return np.log(self.S(R, F))

    def S(self, R, F):
        try:
            len(R)
        except TypeError:
            R = np.array([R])
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
        if F == 0:
            mean = self._m0
            var = self._V0
        elif F == 1:
            mean = self._m1
            var = self._V1
        
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
        N = scipy.stats.norm(self._m0 - self._m1, np.sqrt(self._V0 + self._V1))
        test = N.cdf(0)

        if test > 0.95:
            return 0
        elif test < 0.05:
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
