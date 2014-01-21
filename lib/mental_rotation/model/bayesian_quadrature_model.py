import numpy as np

import matplotlib.pyplot as plt
import logging
import scipy.stats
import scipy.optimize as optim
import pickle

from bayesian_quadrature import BQ
from gp import PeriodicKernel

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

    _iter = 720
    _scale = 100
    _params = ['h', 'w']

    def __init__(self, *args, **kwargs):
        super(BayesianQuadratureModel, self).__init__(*args, **kwargs)

        self.bq_opts = {}
        self.bq_opts['candidate_thresh'] = self.opts['step'] / 2.
        self.bq_opts['kernel'] = PeriodicKernel
        self.bq_opts['optim_method'] = 'Powell'
        self.bq_opts['x_mu'] = np.pi
        self.bq_opts['x_var'] = 10.0
        self.bq_opts['n_candidate'] = 20

        self.bqs = {}
        self._m0 = None
        self._V0 = None
        self._m1 = None
        self._V1 = None

    def _check_done(self):
        hyp = self.hypothesis_test()
        if hyp == 0 or hyp == 1:
            self.status = 'done'
            return True
        return False

    def _add_observation(self, R, F):
        self.model['R'].value = R
        self.model['F'].value = F
        logger.debug(
            "F = %s, R = %s", 
            float(self.model['F'].value),
            self.model['R'].value)

        S = self._scale * np.exp(self.model['log_S'].logp)
        bq = self.bqs[F]
        bq.add_observation(R, S)
        if bq.x_s.size > 1:
            self._fit_hypers(F)

    def _get_actions(self):
        R = self.model['R'].value
        F = float(self.model['F'].value)
        
        actions = [
            (1 - F, 0),
            (1 - F, R),
            (F, 0),
            (F, R + np.abs(self._random_step())),
            (F, R - np.abs(self._random_step()))
        ]

        actions = np.array(sorted(actions))
        actions[:, 1] = self._unwrap(actions[:, 1])
        return actions

    def _eval_actions(self, n=50):
        actions = self._get_actions()

        loss = np.empty(actions.shape[0])
        F0 = actions[:, 0] == 0
        F1 = actions[:, 0] == 1
        x_a0 = np.array(actions[F0, 1])
        x_a1 = np.array(actions[F1, 1])

        funs = [
            lambda: self.bqs[0].expected_squared_mean(x_a0),
            self.bqs[0].Z_mean,
            self.bqs[0].Z_var
        ]
        # l0, m0, V0 = self.bqs[0].marginalize(
        #     funs, n, params=self._params)
        l0, m0, V0 = [np.array([f()]) for f in funs]

        funs = [
            lambda: self.bqs[1].expected_squared_mean(x_a1),
            self.bqs[1].Z_mean,
            self.bqs[1].Z_var
        ]
        # l1, m1, V1 = self.bqs[1].marginalize(
        #     funs, n, params=self._params)
        l1, m1, V1 = [np.array([f()]) for f in funs]

        self._m0 = max(0, m0.mean())
        self._V0 = max(0, V0.mean())
        self._m1 = max(0, m1.mean())
        self._V1 = max(0, V1.mean())

        loss[F0] = (V0 + m0 ** 2 - l0.T).mean(axis=1) + self._V1
        loss[F1] = (V1 + m1 ** 2 - l1.T).mean(axis=1) + self._V0

        return actions, loss

    def _fit_hypers(self, F, ntry=10):
        p0_tl = [self.bqs[F].gp_log_l.get_param(p) for p in self._params]
        p0_l = [self.bqs[F].gp_l.get_param(p) for p in self._params]
        p0 = np.array(p0_tl + p0_l)
        
        logpdf = self.bqs[F]._make_llh_params(self._params)
        def f(x):
            llh = logpdf(x)
            if llh > -np.inf:
                if self.bqs[F].gp_log_l.get_param('w') > (np.pi / 4.):
                    return -np.inf
                if self.bqs[F].gp_l.get_param('w') > (np.pi / 4.):
                    return -np.inf
            return llh
                
        for i in xrange(ntry):
            logger.debug(
                "Fitting parameters %s for F=%d, attempt %d", 
                self._params, F, i+1)

            res = optim.minimize(
                fun=lambda x: -f(x),
                x0=p0,
                method='Powell')

            if f(res['x']) > MIN:
                p0 = res['x']
                break

            if f(p0) > MIN:
                break

            p0 = np.abs(np.random.randn(len(self._params)))

        if f(p0) < MIN:
            raise RuntimeError("couldn't find good parameters")

    def _init_bq(self, F):
        Ri = np.array([self.model['R'].value])
        Si = self._scale * np.array([np.exp(self.model['log_S'].logp)])
        
        logger.debug("Initializing BQ object for F=%d", F)
        self.bqs[F] = BQ(Ri, Si, **self.bq_opts)
        self.bqs[F].init(
            params_tl=(5, np.pi / 8., 1.0, 0.0),
            params_l=(25, np.pi / 8., 1.0, 0.0))

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
        return self.bqs[F].l_mean(R) / self._scale

    ##################################################################
    # Estimated Z

    def log_Z(self, F):
        Z = self.Z(F)
        out = np.empty_like(Z)
        out[Z == 0] = -np.inf
        out[Z != 0] = np.log(Z[Z != 0])
        return out

    def Z(self, F):
        if F == 0:
            mean = self._m0 / self._scale
            var = self._V0 / (self._scale ** 2)
        elif F == 1:
            mean = self._m1 / self._scale
            var = self._V1 / (self._scale ** 2)
        
        lower = mean - 1.96 * np.sqrt(var)
        upper = mean + 1.96 * np.sqrt(var)
        
        out = np.array([mean, lower, upper])
        out[out < 0] = 0
        return out
    
    ##################################################################
    # Plotting methods

    def plot_bq(self, F, f_S=None):
        if f_S is not None:
            f_l = lambda x: self._scale * f_S(x)
        else:
            f_l = None

        self.bqs[F].plot(f_l=f_l, xmin=-np.pi, xmax=np.pi)

    def plot_all(self, f_S0=None, f_S1=None):
        fig, axes = plt.subplots(2, 3)

        if f_S0 is not None:
            f_l0 = lambda x: self._scale * f_S0(x)
        else:
            f_l0 = None

        if f_S1 is not None:
            f_l1 = lambda x: self._scale * f_S1(x)
        else:
            f_l1 = None

        xmin = -np.pi
        xmax = np.pi

        self.bqs[0].plot_gp_log_l(axes[0, 0], f_l=f_l0, xmin=xmin, xmax=xmax)
        self.bqs[0].plot_gp_l(axes[0, 1], f_l=f_l0, xmin=xmin, xmax=xmax)
        self.bqs[0].plot_l(axes[0, 2], f_l=f_l0, xmin=xmin, xmax=xmax)

        self.bqs[1].plot_gp_log_l(axes[1, 0], f_l=f_l1, xmin=xmin, xmax=xmax)
        self.bqs[1].plot_gp_l(axes[1, 1], f_l=f_l1, xmin=xmin, xmax=xmax)
        self.bqs[1].plot_l(axes[1, 2], f_l=f_l1, xmin=xmin, xmax=xmax)

        ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes[:, 1:].flat])
        ymin = min(ymins)
        ymax = max(ymaxs)
        for ax in axes[:, 1:].flat:
            ax.set_ylim(ymin, ymax)

        ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes[:, 0]])
        ymin = min(ymins)
        ymax = max(ymaxs)
        for ax in axes[:, 0]:
            ax.set_ylim(ymin, ymax)

        fig.set_figwidth(14)
        fig.set_figheight(7)

    def plot(self, ax, F, f_S=None, color0='k', color=None):
        lines = {}

        R = np.linspace(-np.pi, np.pi, 1000)
        if f_S is not None:
            S = f_S(R, F)
            lines['truth'] = ax.plot(R, S, '-', lw=2, color=color0)

        Fi = self.F_i == F
        if not Fi.any():
            return lines

        if color is None:
            if F == 0:
                color = 'r'
            elif F == 1:
                color = 'b'

        S = self.S(R, F)
        Ss = np.sqrt(self.bqs[F].l_var(R)) / self._scale
        lower = S - Ss
        upper = S + Ss

        lines['stddev'] = ax.fill_between(R, lower, upper, color=color, alpha=0.3)
        lines['approx'] = ax.plot(R, S, '-', lw=2, color=color)

        Ri = self.R_i[Fi]
        Si = self.S_i[Fi]
        lines['points'] = ax.plot(Ri, Si, 'o', markersize=5, color=color)

        return lines

    ##################################################################
    # Misc

    def hypothesis_test(self):
        m0 = self._m0 / self._scale
        V0 = self._V0 / (self._scale ** 2)
        m1 = self._m1 / self._scale
        V1 = self._V1 / (self._scale ** 2)

        N0 = scipy.stats.norm(m0, np.sqrt(V0))
        if N0.cdf(0) > 0.025:
            return None

        N1 = scipy.stats.norm(m1, np.sqrt(V1))
        if N1.cdf(0) > 0.025:
            return None

        N = scipy.stats.norm(m0 - m1, np.sqrt(V0 + V1))
        test = N.cdf(0)

        if test > 0.975:
            return 1
        elif test < 0.025:
            return 0
        else:
            return None

    @property
    def log_lh_h0(self):
        m = self._m0 / self._scale
        s = np.sqrt(self._V0) / self._scale
        lower = m - 1.96 * s
        upper = m + 1.96 * s
        return np.array([m, lower, upper])

    @property
    def log_lh_h1(self):
        m = self._m1 / self._scale
        s = np.sqrt(self._V1) / self._scale
        lower = m - 1.96 * s
        upper = m + 1.96 * s
        return np.array([m, lower, upper])

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

    ##################################################################
    # Copying/Saving

    def __getstate__(self):
        state = super(BayesianQuadratureModel, self).__getstate__()
        state['_m0'] = self._m0
        state['_V0'] = self._V0
        state['_m1'] = self._m1
        state['_V1'] = self._V1
        state['bqs'] = pickle.dumps(self.bqs)
        state['bq_opts'] = pickle.dumps(self.bq_opts)
        return state

    def __setstate__(self, state):
        super(BayesianQuadratureModel, self).__setstate__(state)
        self._m0 = state['_m0']
        self._V0 = state['_V0']
        self._m1 = state['_m1']
        self._V1 = state['_V1']
        self.bqs = pickle.loads(state['bqs'])
        self.bq_opts = pickle.loads(state['bq_opts'])
