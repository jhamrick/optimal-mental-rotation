import pymc
import numpy as np
import scipy
import matplotlib.pyplot as plt
import snippets.graphing as sg

from .. import Stimulus2D


class BaseModel(pymc.Sampler):

    def __init__(self, X_a, X_b, R_mu, R_kappa, S_sigma):

        @pymc.stochastic(observed=True, trace=False)
        def Xa(value=X_a.vertices):
            return self._prior(value)

        @pymc.stochastic(observed=True, trace=False)
        def Xb(value=X_b.vertices):
            return self._prior(value)

        R = pymc.CircVonMises("R", R_mu, R_kappa, value=0)

        @pymc.deterministic
        def Xr(Xa=Xa, R=R):
            Xr = Xa.copy()
            Stimulus2D._rotate(Xr, np.degrees(R))
            return Xr

        @pymc.potential
        def logS(Xb=Xb, Xr=Xr):
            return self._log_similarity(Xb, Xr, S_sigma)

        vars = {
            'Xa': Xa,
            'Xb': Xb,
            'R': R,
            'Xr': Xr,
            'logS': logS
        }
        name = type(self).__name__
        super(BaseModel, self).__init__(input=vars, name=name)

        self.model = vars
        self._funs_to_tally['logS'] = self.model['logS'].get_logp
        self._funs_to_tally['logp'] = lambda: self.logp

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

    @classmethod
    def _prior(cls, X):
        # the beginning is the same as the end, so ignore the last vertex
        n = X.shape[0] - 1
        # n points picked at random angles around the circle
        log_pangle = -np.log(2*np.pi) * n
        # random radii between 0 and 1
        radius = 1
        log_pradius = -np.log(radius) * n
        # number of possible permutations of the points
        log_pperm = np.log(scipy.misc.factorial(n))
        # put it all together
        logp = log_pperm + log_pangle + log_pradius
        return logp

    @classmethod
    def _log_similarity(cls, X0, X1, S_sigma):
        """Computes the similarity between sets of vertices `X0` and `X1`."""
        # number of points and number of dimensions
        n, D = X0.shape
        # covariance matrix
        Sigma = np.eye(D) * S_sigma
        invSigma = np.eye(D) * (1. / S_sigma)
        # iterate through all permutations of the vertices -- but if
        # two vertices are connected, they are next to each other in
        # the list (or on the ends), so we really only need to cycle
        # through 2n orderings (once for the original ordering, and
        # once for the reverse)
        e = np.empty(2*n)
        for i in xrange(n):
            idx = np.arange(i, i+n) % n
            d = X0 - X1[idx]
            e[i] = -0.5 * np.sum(np.dot(d, invSigma) * d)
        for i in xrange(n):
            idx = np.arange(i, i+n)[::-1] % n
            d = X0 - X1[idx]
            e[i+n] = -0.5 * np.sum(np.dot(d, invSigma) * d)
        # constants
        Z0 = (D / 2.) * np.log(2 * np.pi)
        Z1 = 0.5 * np.linalg.slogdet(Sigma)[1]
        # overall similarity, marginalizing out order
        logS = np.log(np.sum(np.exp(e + Z0 + Z1 - np.log(n))))
        return logS
