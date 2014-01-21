import numpy as np
import scipy.optimize as optim
import logging

from . import BaseModel

logger = logging.getLogger("mental_rotation.model.oracle")


class OracleModel(BaseModel):

    _iter = 720

    def __init__(self, *args, **kwargs):
        super(OracleModel, self).__init__(*args, **kwargs)
        self.direction = None
        self.target = None

    ##################################################################
    # Sampling

    def _solve(self):
        Xa = self.model['Xa'].value
        Xb = self.model['Xb'].value
        F = np.array([[-1, 0], [0, 1]], dtype=float)

        if np.allclose(Xa, Xb):
            h = 0
            theta = 0

        elif np.allclose(np.dot(Xa, F.T), Xb):
            h = 1
            theta = 0

        else:
            R = np.round(np.dot(np.linalg.pinv(Xa), Xb).T, decimals=9)

            # rotated 180 degrees
            if np.allclose(R, np.array([[-1.0, 0.0], [0.0, -1.0]])):
                h = 0
                theta = np.pi

            elif np.sign(R[0, 1]) == np.sign(R[1, 0]):
                h = 1
                R = np.dot(np.linalg.pinv(np.dot(Xa, F.T)), Xb).T
                costheta = R[0, 0]
                sintheta = R[1, 0]
                theta = np.arctan2(sintheta, costheta)

            else:
                h = 0
                costheta = R[0, 0]
                sintheta = R[1, 0]
                theta = np.arctan2(sintheta, costheta)

        return theta, h

    def draw(self):
        if self._current_iter == 0:
            R, F = self._solve()
            self.model['R'].value = 0
            self.model['F'].value = F
            self.target = R
            self.direction = np.sign(self._unwrap(R))
            return

        R = self.model['R'].value
        if np.abs(self._unwrap(R - self.target)) < self.opts['step']:
            self.model['R'].value = self.target
            self.status = 'done'

        else:
            step = self.direction * np.abs(self._random_step())
            self.model['R'].value = R + step

    ##################################################################
    # Plotting methods
    def plot(self, ax, F=None):
        Fi = self.F_i

        if F is None or F == 0:
            Ri0 = self.R_i[Fi == 0]
            Si0 = self.S_i[Fi == 0]
            ax.plot(Ri0, Si0, 'r-', lw=2)
            ax.plot(Ri0, Si0, 'ro', markersize=5)

        if F is None or F == 1:
            Ri1 = self.R_i[Fi == 1]
            Si1 = self.S_i[Fi == 1]
            ax.plot(Ri1, Si1, 'b-', lw=2)
            ax.plot(Ri1, Si1, 'bo', markersize=5)

    ##################################################################
    # Copying/Saving

    def __getstate__(self):
        state = super(OracleModel, self).__getstate__()
        state['direction'] = self.direction
        state['target'] = self.direction
        return state

    def __setstate__(self, state):
        super(OracleModel, self).__setstate__(state)
        self.direction = state['direction']
        self.target = state['target']
