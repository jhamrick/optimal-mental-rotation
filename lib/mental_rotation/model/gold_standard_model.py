import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import animation

from . import BaseModel
from .. import Stimulus2D

logger = logging.getLogger("mental_rotation.model.gs")


class GoldStandardModel(BaseModel):

    _iter = 362

    def draw(self):
        i = self._current_iter % (self._iter / 2)
        self.model['F'].value = int(self._current_iter >= (self._iter / 2))

        if i == 0:
            self.model['R'].value = -np.pi
        else:
            self.model['R'].value = self.model['R'].value + np.radians(2)

    ##################################################################
    # The estimated S function

    def log_S(self, R, F):
        return np.log(self.S(R, F))

    def S(self, R, F):
        R_i = self.R_i
        F_i = self.F_i
        match = F_i == F
        ix = np.argsort(R_i[match])
        Ri = R_i[match][ix]
        Si = self.S_i[match][ix]
        S = np.interp(self._unwrap(R), Ri, Si)
        return S

    ##################################################################
    # Plotting methods

    def plot(self, ax, F, f_S=None, color0=None, color='k'):
        if f_S is not None:
            logger.warn("f_S is not used by this function")
        if color0 is not None:
            logger.warn("color0 is not used by this function")

        R = np.linspace(-np.pi, np.pi, 1000)
        S = self.S(R, F)

        lines = {}
        lines['approx'] = ax.plot(R, S, '-', color=color, lw=2)
        return lines

    def animate(self, interval=1):
        R_i = self.R_i
        F_i = self.F_i
        S_i = self.S_i

        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax3 = plt.subplot2grid((2, 3), (1, 2))
        fig = plt.gcf()

        line0, = ax1.plot(
            [], [], 'r-', lw=2,
            label="same ($h=0$)")
        line1, = ax1.plot(
            [], [], 'b-', lw=2,
            label="flipped ($h=1$)")
        curr_line, = ax1.plot([], [], 'k-', alpha=0.5)
        Xa, = ax2.plot([], [], 'k-', alpha=0.2, lw=2)
        Xr, = ax2.plot([], [], 'k-', lw=2)
        Xb, = ax3.plot([], [], 'k-', lw=2)
        lines = [line0, line1, curr_line, Xa, Xr, Xb]

        ymin = 0
        ymax = 0.14

        ax1.legend(loc="upper left", fontsize=14, frameon=False)
        ax1.set_xlim(-np.pi, np.pi)
        ax1.set_xticks([-np.pi, -np.pi / 2., 0.0, np.pi / 2.0, np.pi])
        ax1.set_xticklabels([-180, -90, 0, 90, 180])
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Rotation", fontsize=16)
        ax1.set_ylabel("Similarity", fontsize=16)

        ax1.spines['top'].set_color('none')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.spines['right'].set_color('none')
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(direction='out')
        ax1.tick_params(axis='both', which='major', labelsize=14)

        for ax in [ax2, ax3]:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.axis([-1, 1, -1, 1])
            ax.set_aspect('equal')
            ax.axis('off')

        fig.set_figwidth(10)
        fig.set_figheight(6)
        plt.subplots_adjust(bottom=0.2)

        def init():
            for line in lines:
                line.set_data([], [])

            v = self.model['Xa'].value.copy()
            X = np.empty((v.shape[0] + 1, 2))
            X[:-1] = v
            X[-1] = v[0]

            Xa.set_data(X[:, 0], X[:, 1])

            v = self.model['Xb'].value.copy()
            X = np.empty((v.shape[0] + 1, 2))
            X[:-1] = v
            X[-1] = v[0]

            Xb.set_data(X[:, 0], X[:, 1])

            return lines

        def plot(i):
            ix = np.argsort(R_i[:(i + 1)])
            ix0 = np.nonzero(F_i[ix] == 0)
            ix1 = np.nonzero(F_i[ix] == 1)
            R0 = R_i[ix][ix0]
            S0 = S_i[ix][ix0]
            R1 = R_i[ix][ix1]
            S1 = S_i[ix][ix1]
            lines[0].set_data(R0, S0)
            lines[1].set_data(R1, S1)

            curr_line.set_data([R_i[i], R_i[i]], [ymin, ymax])

            v = self.model['Xa'].value.copy()
            if F_i[i] == 1:
                Stimulus2D._flip(v, np.array([0, 1]))
            Stimulus2D._rotate(v, np.degrees(R_i[i]))
            X = np.empty((v.shape[0] + 1, 2))
            X[:-1] = v
            X[-1] = v[0]

            Xr.set_data(X[:, 0], X[:, 1])

            return lines

        anim = animation.FuncAnimation(
            fig, plot,
            init_func=init,
            frames=len(R_i),
            interval=interval,
            blit=False)

        return anim
