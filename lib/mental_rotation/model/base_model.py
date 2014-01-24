import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats

from path import path
from . import model


class BaseModel(object):

    _iter = None

    def __init__(self, Xa, Xb, **opts):
        self.opts = opts.copy()
        self._make_model(Xa, Xb)
        self._current_iter = None
        self._traces = None
        self.status = "ready"

    def _make_model(self, Xa, Xb):
        self.model = {}
        self.model['Xa'] = model.Xi('Xa', np.array(Xa))
        self.model['Xb'] = model.Xi('Xb', np.array(Xb))
        self.model['F'] = model.F()
        self.model['R'] = model.R()
        self.model['Xr'] = model.Xr(
            self.model['Xa'], self.model['R'], self.model['F'])
        self.model['log_S'] = model.log_S(
            self.model['Xb'], self.model['Xr'], self.opts['S_sigma'])

        self._log_const = model.log_const(
            self.model['Xa'].value.shape[0],
            self.model['Xa'].value.shape[1],
            self.opts['S_sigma'])

    def _init_traces(self):
        n = self._iter
        self._traces = {}
        self._traces['F'] = np.empty(n)
        self._traces['R'] = np.empty(n)
        self._traces['Xr'] = np.empty((n,) + self.model['Xr'].value.shape)
        self._traces['log_S'] = np.empty(n)

    def trace(self, var):
        return self._traces[var][:self._current_iter]

    def _finish(self):
        i = self._current_iter
        for v in self._traces:
            self._traces[v] = self._traces[v][:i]

    def tally(self):
        i = self._current_iter
        self._traces['F'][i] = self.model['F'].value
        self._traces['R'][i] = self.model['R'].value
        self._traces['Xr'][i] = self.model['Xr'].value.copy()
        self._traces['log_S'][i] = self.model['log_S'].logp

    def sample(self):
        if self.status == "done":
            return

        if self.status == "ready":
            self._current_iter = 0
            self._init_traces()

        self.status = "running"

        try:
            self.loop()
        except KeyboardInterrupt:
            self.status = "paused"

        if self.status == "paused":
            self._restore(self._current_iter - 1)

        elif self._current_iter == self._iter:
            self.status = "done"

        if self.status == "done":
            self._finish()
            
    def loop(self):
        while self._current_iter < self._iter and self.status == 'running':
            self.draw()
            self.tally()
            self._current_iter += 1

    def draw(self):
        pass

    def _restore(self, i):
        self.model['F'].value = self._traces['F'][i]
        self.model['R'].value = self._traces['R'][i]

    ##################################################################
    # Sampled R_i

    @property
    def R_i(self):
        R = self.trace('R')
        return R

    ##################################################################
    # Sampled F_i

    @property
    def F_i(self):
        F = self.trace('F')
        return F

    ##################################################################
    # Sampled S_i and the estimated S

    @property
    def log_S_i(self):
        log_S = self.trace('log_S')
        return log_S

    @property
    def S_i(self):
        return np.exp(self.log_S_i)

    def log_S(self, R, F):
        raise NotImplementedError

    def S(self, R, F):
        raise NotImplementedError

    ##################################################################
    # Log likelihoods for each hypothesis

    @property
    def log_lh_h0(self):
        Fi = self.F_i == 0
        if Fi.any():
            log_S0 = self.log_S_i[Fi].max()
        else:
            log_S0 = -np.inf
        return log_S0

    @property
    def log_lh_h1(self):
        Fi = self.F_i == 1
        if Fi.any():
            log_S1 = self.log_S_i[Fi].max()
        else:
            log_S1 = -np.inf
        return log_S1

    def hypothesis_test(self):
        llh0 = self.log_lh_h0
        llh1 = self.log_lh_h1

        if np.isclose(llh0, llh1):
            return None
        elif llh0 > llh1:
            return 0
        else:
            return 1

    ##################################################################
    # Plotting 

    def plot(self, ax, F, f_S=None, color0='k', color=None):
        lines = {}

        if f_S is not None:
            R = np.linspace(-np.pi, np.pi, 1000)
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

        Ri = self.R_i[Fi]
        Si = self.S_i[Fi]
        ii = np.argsort(Ri)

        lines['approx'] = ax.plot(Ri[ii], Si[ii], '-', lw=2, color=color)
        lines['points'] = ax.plot(Ri[ii], Si[ii], 'o', markersize=5, color=color)

        return lines

    def plot_trace(self, ax, legend=True):
        Fi = self.F_i
        Si = self.log_S_i - self._log_const
        Si = (Si - Si.min() + 1) * 10
        Ri = self.R_i
        ti = np.arange(Ri.size)
        ci = np.empty((Ri.size, 3))
        ci[Fi == 0] = np.array([1, 0, 0])
        ci[Fi == 1] = np.array([0, 0, 1])

        ax.plot(ti, Ri, 'k-')
        ax.scatter(ti, Ri, c=ci, s=Si, edgecolor=ci)
    
        ax.set_xlabel("Action #", fontsize=14)
        ax.set_xlim(0, Ri.size - 1)
 
        ax.set_ylim(np.pi+0.4, -np.pi-0.4)
        ax.set_yticks([np.pi, np.pi / 2., 0, -np.pi / 2., -np.pi])
        ax.set_yticklabels([180, 90, 0, -90, -180])

        if legend:
            p0 = plt.Rectangle((0, 0), 1, 1, fc="r", ec="r")
            p1 = plt.Rectangle((0, 0), 1, 1, fc="b", ec="b")
            ax.legend(
                [p0, p1], ["same", "flipped"], 
                frameon=False, numpoints=1, fontsize=12)

    ##################################################################
    # Misc

    def print_stats(self):
        print "log LH(h0) = %f" % self.log_lh_h0
        print "log LH(h1) = %f" % self.log_lh_h1

        llr = self.log_lh_h0 - self.log_lh_h1
        print "log LH(h0) / LH(h1) = %f" % llr
        if llr < 0: # pragma: no cover
            print "--> STOP and accept hypothesis 1 (flipped)"
        elif llr > 0: # pragma: no cover
            print "--> STOP and accept hypothesis 0 (same)"
        else: # pragma: no cover
            print "--> UNDECIDED"

    def _wrap(self, x):
        return x % (2 * np.pi)

    def _unwrap(self, x):
        x_ = x % (2 * np.pi)
        try:
            x_[x_ > np.pi] -= 2 * np.pi
        except (TypeError, IndexError):
            if x_ > np.pi:
                x_ -= 2 * np.pi
        return x_

    def _random_step(self):
        step = self.opts['step']
        x = scipy.stats.norm.rvs(0, np.sqrt(step / 2.))
        R = np.clip(x, -step, step)
        # R = np.random.uniform(-step, step)
        return R


    ##################################################################
    # Copying/Saving
    
    def __getstate__(self):
        state = {}
        state['opts'] = self.opts
        state['Xa'] = self.model['Xa'].value.tolist()
        state['Xb'] = self.model['Xb'].value.tolist()
        state['_current_iter'] = self._current_iter
        state['_traces'] = self._traces
        state['status'] = self.status
        return state

    def __setstate__(self, state):
        self.opts = state['opts']
        self._make_model(state['Xa'], state['Xb'])
        self._current_iter = state['_current_iter']
        self._traces = state['_traces']
        if self._current_iter is not None:
            self._restore(self._current_iter - 1)

        self.status = state['status']

    def save(self, loc, force=False):
        loc = path(loc)
        if loc.exists() and not force:
            raise IOError("path %s already exists" % loc.abspath())
        elif loc.exists() and force:
            loc.rmtree_p()

        loc.mkdir_p()

        state = self.__getstate__()
        traces = state['_traces']
        del state['_traces']

        with open(loc.joinpath("state.json"), "w") as fh:
            json.dump(state, fh)

        if traces is not None:
            tloc = loc.joinpath("traces")
            tloc.mkdir_p()

            for name in traces:
                trace = traces[name]
                np.save(tloc.joinpath(name + ".npy"), trace)

    @classmethod
    def load(cls, loc):
        loc = path(loc)
        if not loc.exists():
            raise IOError("path does not exist: %s" % loc.abspath())

        with open(loc.joinpath("state.json"), "r") as fh:
            state = json.load(fh)

        tloc = loc.joinpath("traces")
        if tloc.exists():
            traces = {}
            for trace in tloc.listdir():
                traces[trace.namebase] = np.load(trace)
            state['_traces'] = traces
        else:
            state['_traces'] = None

        model = cls.__new__(cls)
        model.__setstate__(state)
        return model
