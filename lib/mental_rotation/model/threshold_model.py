import numpy as np

from . import BaseModel


class ThresholdModel(BaseModel):

    _iter = 720

    def __init__(self, *args, **kwargs):
        super(ThresholdModel, self).__init__(*args, **kwargs)
        self.direction = None
        self._thresh = 0.6 * np.exp(self._log_const)

    ##################################################################
    # Sampling

    def draw(self):
        if self._current_iter == 0:
            self.model['R'].value = 0
            self.model['F'].value = 0
            S0 = np.exp(self.model['log_S'].logp)

            self.tally()
            self._current_iter += 1
            self.model['F'].value = 1
            S1 = np.exp(self.model['log_S'].logp)

            if S0 > S1:
                self.tally()
                self._current_iter += 1
                self.model['F'].value = 0

            if np.exp(self.model['log_S'].logp) > self._thresh:
                self.status = 'done'

            return

        R = self.model['R'].value
        if self.direction is None:
            step = self._random_step()
            self.direction = np.sign(step)
            self.model['R'].value = step

            new_log_S = self.model['log_S'].logp
            if np.exp(new_log_S) > self._thresh:
                self.status = 'done'
                return

        else:
            step = self.direction * np.abs(self._random_step())
            self.model['R'].value = R + step
            new_log_S = self.model['log_S'].logp

            if np.exp(new_log_S) > self._thresh:
                self.status = 'done'
                return

            if (R > 0 and (R + step) < 0) or (R < 0 and (R + step) > 0):
                self.tally()
                self._current_iter += 1
                self.direction = None
                self.model['R'].value = 0
                self.model['F'].value = 1 - self.model['F'].value

            else: # pragma: no cover
                pass

    ##################################################################
    # Copying/Saving

    def __getstate__(self):
        state = super(ThresholdModel, self).__getstate__()
        state['direction'] = self.direction
        state['_thresh'] = self._thresh
        return state

    def __setstate__(self, state):
        super(ThresholdModel, self).__setstate__(state)
        self.direction = state['direction']
        self._thresh = state['_thresh']
