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
        self.model['logS'] = model.make_logS(
            self.model['Xb'], self.model['Xr'], S_sigma)

        self._prior = model.prior
        self._log_similarity = model.log_similarity

        name = type(self).__name__
        super(BaseModel, self).__init__(input=self.model, name=name)

        self._funs_to_tally['logS'] = self.model['logS'].get_logp
        self._funs_to_tally['logp'] = self.get_logp

    def get_logp(self):
        return self.logp

    def _loop(self):
        self.tally()
        super(BaseModel, self)._loop()

    def integrate(self):
        raise NotImplementedError

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
            xn[xn < 0] += 2*np.pi
            ax.plot(xn[ix], y[ix], 'k-', label="actual", linewidth=2)
        if xi is not None:
            xin = xi.copy()
            xin[xin < 0] += 2*np.pi
            ax.plot(xi, yi, 'ro', label="samples")

        if xo is not None:
            ix = np.argsort(xo)
            xon = xo.copy()
            xon[xon < 0] += 2*np.pi

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
        if opt['xlabel']:
            ax.set_xlabel(opt['xlabel'])
        if opt['ylabel']:
            ax.set_ylabel(opt['ylabel'])

        if opt['legend']:
            ax.legend(loc=0, fontsize=12, frameon=False)

    def plot(self, ax):
        raise NotImplementedError

    @property
    def S(self):
        raise NotImplementedError

    @property
    def R_i(self):
        R = self.trace('R')[:]
        R[R < 0] += 2*np.pi
        return R

    @property
    def S_i(self):
        S = np.exp(self.trace('logS')[:])
        return S

    @property
    def p_i(self):
        p = np.exp(self.trace('logp')[:])
        return p
