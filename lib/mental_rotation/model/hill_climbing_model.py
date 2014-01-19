import numpy as np

from . import BaseModel


class HillClimbingModel(BaseModel):

    _iter = 720

    def __init__(self, *args, **kwargs):
        super(HillClimbingModel, self).__init__(*args, **kwargs)
        self.direction = None

    ##################################################################
    # Sampling

    def draw(self):
        if self._current_iter == 0:
            self.model['R'].value = 0
            self.model['F'].value = 0
            return

        R = self.model['R'].value
        F = self.model['F'].value
        log_S = self.model['log_S'].logp

        if self.direction is None:
            step = self._random_step()
            self.direction = np.sign(step)
            self.model['R'].value = step

            new_log_S = self.model['log_S'].logp
            if new_log_S < log_S:
                self.tally()
                self._current_iter += 1
                self.direction *= -1
                self.model['R'].value = 0

        else:
            step = self.direction * np.abs(self._random_step())
            self.model['R'].value = R + step
            new_log_S = self.model['log_S'].logp

            if new_log_S < log_S:
                if F == 0:
                    self.tally()
                    self._current_iter += 1
                    self.direction = None
                    self.model['R'].value = 0
                    self.model['F'].value = 1
                else:
                    self.status = 'done'

            else: # pragma: no cover
                pass

    ##################################################################
    # Plotting methods

    def plot(self, ax):
        Fi = self.F_i

        Ri0 = self.R_i[Fi == 0]
        Si0 = self.S_i[Fi == 0]

        Ri1 = self.R_i[Fi == 1]
        Si1 = self.S_i[Fi == 1]

        ax.plot(Ri0, Si0, 'ro', markersize=7)
        ax.plot(Ri1, Si1, 'bo', markersize=7)

    ##################################################################
    # Copying/Saving

    def __getstate__(self):
        state = super(HillClimbingModel, self).__getstate__()
        state['direction'] = self.direction
        return state

    def __setstate__(self, state):
        super(HillClimbingModel, self).__setstate__(state)
        self.direction = state['direction']
