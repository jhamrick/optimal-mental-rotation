import numpy as np
import snippets.graphing as sg
import json

from path import path
from . import model
from .. import config


class BaseModel(object):

    def __init__(self, Xa, Xb, name=None, **opts):

        self.opts = {
            'R_mu': config.getfloat("model", "R_mu"),
            'R_kappa': config.getfloat("model", "R_kappa"),
            'S_sigma': config.getfloat("model", "S_sigma"),
            'step': config.getfloat("model", "step")
        }
        self.opts.update(opts)

        self._make_model(Xa, Xb)

        self._current_iter = 0
        self._iter = 0
        self._traces = None

        self.status = "ready"

    def _make_model(self, Xa, Xb):
        self.model = {}
        self.model['Xa'] = model.Xi('Xa', np.array(Xa))
        self.model['Xb'] = model.Xi('Xb', np.array(Xb))
        self.model['F'] = model.F()
        self.model['R'] = model.R(
            self.opts['R_mu'], self.opts['R_kappa'])
        self.model['Xr'] = model.Xr(
            self.model['Xa'], self.model['R'], self.model['F'])
        self.model['log_S'] = model.log_S(
            self.model['Xb'], self.model['Xr'], self.opts['S_sigma'])
        self.model['log_dZ_dR'] = model.log_dZ_dR(
            self.model['log_S'], self.model['R'], self.model['F'])

    def _init_traces(self):
        n = self._iter
        self._traces = {}
        self._traces['F'] = np.empty(n)
        self._traces['R'] = np.empty(n)
        self._traces['Xr'] = np.empty((n,) + self.model['Xr'].value.shape)
        self._traces['log_S'] = np.empty(n)
        self._traces['log_dZ_dR'] = np.empty(n)

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
        self._traces['log_dZ_dR'][i] = self.model['log_dZ_dR'].logp

    def sample(self, niter):
        if self.status == "done":
            return

        if self.status == "ready":
            self._iter = niter
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

        elif self.status == "done":
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
    # Sampled dZ_dR (which is just S_i*p(R_i)) and full estimate of Z

    @property
    def log_dZ_dR_i(self):
        log_p = self.trace('log_dZ_dR')
        return log_p

    @property
    def dZ_dR_i(self):
        return np.exp(self.log_dZ_dR_i)

    def log_dZ_dR(self, R, F):
        raise NotImplementedError

    def dZ_dR(self, R, F):
        raise NotImplementedError

    def log_Z(self, F):
        raise NotImplementedError

    def Z(self, F):
        raise NotImplementedError

    ##################################################################
    # Log likelihoods for each hypothesis

    @property
    def log_lh_h0(self):
        log_Z = self.log_Z(0)
        p_Xa = self.model['Xa'].logp
        return log_Z + p_Xa

    @property
    def log_lh_h1(self):
        log_Z = self.log_Z(1)
        p_Xa = self.model['Xa'].logp
        return log_Z + p_Xa

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

    def plot(self, ax):
        raise NotImplementedError

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

    ##################################################################
    # Copying/Saving
    
    def __getstate__(self):
        state = {}
        state['opts'] = self.opts
        state['Xa'] = self.model['Xa'].value.tolist()
        state['Xb'] = self.model['Xb'].value.tolist()
        state['_current_iter'] = self._current_iter
        state['_iter'] = self._iter
        state['_traces'] = self._traces
        state['status'] = self.status
        return state

    def __setstate__(self, state):
        self.opts = state['opts']
        self._make_model(state['Xa'], state['Xb'])
        self._current_iter = state['_current_iter']
        self._iter = state['_iter']
        self._traces = state['_traces']
        self.status = state['status']

    def save(self, loc, force=False):
        loc = path(loc)
        if loc.exists() and not force:
            raise IOError("path %s already exists" % loc.abspath())
        elif loc.exists() and force:
            loc.rmdir_p()

        loc.mkdir_p()
        tloc = loc.joinpath("traces")
        tloc.mkdir_p()

        state = self.__getstate__()
        traces = state['_traces']
        del state['_traces']

        with open(loc.joinpath("state.json"), "w") as fh:
            json.dump(state, fh)

        for name in traces:
            trace = traces[name]
            np.save(tloc.joinpath(name + ".npy"), trace)

    @classmethod
    def load(cls, loc):
        loc = path(loc)
        if not loc.exists():
            raise IOError("path does not exist: %s" % loc.abspath())

        tloc = loc.joinpath("traces")
        if not tloc.exists():
            raise IOError("trace path does not exist: %s" % tloc.abspath())

        with open(loc.joinpath("state.json"), "r") as fh:
            state = json.load(fh)

        traces = {}
        for trace in tloc.listdir():
            traces[trace.namebase] = np.load(trace)
        state['_traces'] = traces

        model = cls.__new__(cls)
        model.__setstate__(state)
        return model
