import pymc
import numpy as np
import snippets.graphing as sg
from path import path

from . import model
from .. import config


class BaseModel(pymc.Sampler):

    def __init__(self, X_a, X_b, name=None, **opts):

        self.opts = {
            'R_mu': config.getfloat("model", "R_mu"),
            'R_kappa': config.getfloat("model", "R_kappa"),
            'S_sigma': config.getfloat("model", "S_sigma"),
            'step': config.getfloat("model", "step")
        }
        self.opts.update(opts)

        self.model = {}
        self.model['Xa'] = model.make_Xi('Xa', X_a)
        self.model['Xb'] = model.make_Xi('Xb', X_b)
        self.model['R'] = model.make_R(
            self.opts['R_mu'], self.opts['R_kappa'])
        self.model['F'] = model.make_F()
        self.model['Xr'] = model.make_Xr(
            self.model['Xa'], self.model['R'], self.model['F'])
        self.model['log_S'] = model.make_log_S(
            self.model['Xb'], self.model['Xr'], self.opts['S_sigma'])

        self._prior = model.prior
        self._log_similarity = model.log_similarity

        if name is None:
            name = type(self).__name__
            db_args = dict(db='ram')
        else:
            db_args = dict(db='hdf5', dbmode='w')
            if path(name + ".hdf5").exists():
                raise IOError(
                    "Database already exists! Use `load` instead to load it.")

        super(BaseModel, self).__init__(
            input=self.model, name=name, **db_args)

        self._funs_to_tally['log_S'] = self.model['log_S'].get_logp
        self._funs_to_tally['log_dZ_dR'] = self.get_log_dZ_dR

    ##################################################################
    # Overwritten PyMC sampling methods

    def _loop(self):
        if self._current_iter == 0:
            self.tally()
            self._current_iter += 1
        super(BaseModel, self)._loop()

    ##################################################################
    # Sampled R_i

    @property
    def R_i(self):
        R = self.trace('R')[:self._current_iter]
        return R

    ##################################################################
    # Sampled F_i

    @property
    def F_i(self):
        F = self.trace('F')[:self._current_iter]
        return F

    ##################################################################
    # Sampled S_i and the estimated S

    @property
    def log_S_i(self):
        log_S = self.trace('log_S')[:self._current_iter]
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

    def get_log_dZ_dR(self):
        p_log_S = self.model['log_S'].logp
        p_R = self.model['R'].logp
        p_F = self.model['F'].logp
        return p_log_S + p_R + p_F

    @property
    def log_dZ_dR_i(self):
        log_p = self.trace('log_dZ_dR')[:self._current_iter]
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

    ##################################################################
    # Plotting methods

    @staticmethod
    def _plot(ax, x, y, xi, yi, xo, yo_mean, yo_var, **kwargs):
        """Plot the original function and the regression estimate.

        """

        opt = {
            'title': None,
            'xlabel': "Rotation ($R$)",
            'ylabel': "Similarity ($S$)",
            'legend': True,
        }
        opt.update(kwargs)

        if x is not None:
            ix = np.argsort(x)
            xn = x.copy()
            ax.plot(xn[ix], y[ix], 'k-', label="actual", linewidth=2)
        if xi is not None:
            ax.plot(xi, yi, 'ro', label="samples")

        if xo is not None:
            ix = np.argsort(xo)

            if yo_var is not None:
                # hack, for if there are zero or negative variances
                yv = np.abs(yo_var)[ix]
                ys = np.zeros(yv.shape)
                ys[yv != 0] = np.sqrt(yv[yv != 0])
                # compute upper and lower bounds
                lower = yo_mean[ix] - ys
                upper = yo_mean[ix] + ys
                ax.fill_between(xo[ix], lower, upper, color='r', alpha=0.25)

            ax.plot(xo[ix], yo_mean[ix], 'r-', label="estimate", linewidth=2)

        # customize x-axis
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2., 0, np.pi / 2., np.pi])
        ax.set_xticklabels(
            [r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$",
             r"$\frac{\pi}{2}$", r"$\pi$"])

        # axis styling
        sg.outward_ticks(ax=ax)
        sg.clear_right(ax=ax)
        sg.clear_top(ax=ax)
        sg.set_scientific(-2, 3, axis='y', ax=ax)

        # title and axis labels
        if opt['title']:
            ax.set_title(opt['title'])
        else: # pragma: no cover
            ax.set_title("")

        if opt['xlabel']:
            ax.set_xlabel(opt['xlabel'])
        else: # pragma: no cover
            ax.set_xlabel("")

        if opt['ylabel']:
            ax.set_ylabel(opt['ylabel'])
        else: # pragma: no cover
            ax.set_ylabel("")

        if opt['legend']:
            ax.legend(loc=0, fontsize=12, frameon=False)
        else: # pragma: no cover
            pass

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
    
    def get_state(self):
        state = super(BaseModel, self).get_state()
        state['opts'] = self.opts
        state['Xa'] = self.Xa.value
        state['Xb'] = self.Xb.value
        return state

    @classmethod
    def load(cls, dbname):
        db = pymc.database.hdf5.load(dbname + ".hdf5")
        state = db.getstate()
        model = cls(state['Xa'], state['Xb'], **state['opts'])
        model.db = db
        model.restore_sampler_state()
        return model

    def save(self):
        self.db.close()
