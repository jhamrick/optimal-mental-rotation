import pymc
import numpy as np
import matplotlib.pyplot as plt
import snippets.graphing as sg

from . import model


class BaseModel(pymc.Sampler):

    def __init__(self, X_a, X_b, R_mu, R_kappa, S_sigma):

        self.model = {}
        self.model['Xa'] = model.make_Xi('Xa', X_a)
        self.model['Xb'] = model.make_Xi('Xb', X_b)
        self.model['R'] = model.make_R(R_mu, R_kappa)
        self.model['Xr'] = model.make_Xr(
            self.model['Xa'], self.model['R'])
        self.model['log_S'] = model.make_log_S(
            self.model['Xb'], self.model['Xr'], S_sigma)

        self._prior = model.prior
        self._log_similarity = model.log_similarity

        name = type(self).__name__
        super(BaseModel, self).__init__(input=self.model, name=name)

        self._funs_to_tally['log_S'] = self.model['log_S'].get_logp
        self._funs_to_tally['log_dZ_dR'] = self.get_log_dZ_dR

    ##################################################################
    # Overwritten PyMC sampling methods

    def _loop(self):
        if self._current_iter == 0:
            self.tally()
        super(BaseModel, self)._loop()

    ##################################################################
    # Sampled R_i

    @property
    def R_i(self):
        R = self.trace('R')[:]
        R[R < 0] += 2 * np.pi
        return R

    ##################################################################
    # Sampled S_i and the estimated S

    @property
    def log_S_i(self):
        log_S = self.trace('log_S')[:]
        return log_S

    @property
    def S_i(self):
        return np.exp(self.log_S_i)

    def log_S(self, R):
        raise NotImplementedError

    def S(self, R):
        raise NotImplementedError

    ##################################################################
    # Sampled dZ_dR (which is just S_i*p(R_i)) and full estimate of Z

    def get_log_dZ_dR(self):
        return self.model['log_S'].logp + self.model['R'].logp

    @property
    def log_dZ_dR_i(self):
        log_p = self.trace('log_dZ_dR')[:]
        return log_p

    @property
    def dZ_dR_i(self):
        return np.exp(self.log_dZ_dR_i)

    def log_dZ_dR(self, R):
        raise NotImplementedError

    def dZ_dR(self, R):
        raise NotImplementedError

    @property
    def log_Z(self):
        raise NotImplementedError

    @property
    def Z(self):
        raise NotImplementedError

    ##################################################################
    # Log likelihoods for each hypothesis

    @property
    def log_lh_h0(self):
        p_Xa = self.model['Xa'].logp
        p_Xb = self.model['Xb'].logp
        return p_Xa + p_Xb

    @property
    def log_lh_h1(self):
        log_Z = self.log_Z
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

        # overall figure settings
        fig = ax.get_figure()
        sg.set_figsize(5, 3, fig=fig)
        plt.subplots_adjust(
            wspace=0.2, hspace=0.3,
            left=0.15, bottom=0.2, right=0.95)

        if x is not None:
            ix = np.argsort(x)
            xn = x.copy()
            xn[xn < 0] += 2 * np.pi
            ax.plot(xn[ix], y[ix], 'k-', label="actual", linewidth=2)
        if xi is not None:
            xin = xi.copy()
            xin[xin < 0] += 2 * np.pi
            ax.plot(xi, yi, 'ro', label="samples")

        if xo is not None:
            ix = np.argsort(xo)
            xon = xo.copy()
            xon[xon < 0] += 2 * np.pi

            if yo_var is not None:
                # hack, for if there are zero or negative variances
                yv = np.abs(yo_var)[ix]
                ys = np.zeros(yv.shape)
                ys[yv != 0] = np.sqrt(yv[yv != 0])
                # compute upper and lower bounds
                lower = yo_mean[ix] - ys
                upper = yo_mean[ix] + ys
                ax.fill_between(xon[ix], lower, upper, color='r', alpha=0.25)

            ax.plot(xon[ix], yo_mean[ix], 'r-', label="estimate", linewidth=2)

        # customize x-axis
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(
            [0, np.pi / 2., np.pi, 3 * np.pi / 2., 2 * np.pi],
            ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])

        # axis styling
        sg.outward_ticks()
        sg.clear_right()
        sg.clear_top()
        sg.set_scientific(-2, 3, axis='y')

        # title and axis labels
        if opt['title']:
            ax.set_title(opt['title'])
        else: # pragma: no cover
            pass

        if opt['xlabel']:
            ax.set_xlabel(opt['xlabel'])
        else: # pragma: no cover
            pass

        if opt['ylabel']:
            ax.set_ylabel(opt['ylabel'])
        else: # pragma: no cover
            pass

        if opt['legend']:
            ax.legend(loc=0, fontsize=12, frameon=False)
        else: # pragma: no cover
            pass

    def plot(self, ax):
        raise NotImplementedError
