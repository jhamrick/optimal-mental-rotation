import numpy as np

from . import BaseModel


class ThresholdModel(BaseModel):

    _iter = 720

    def __init__(self, *args, **kwargs):
        super(ThresholdModel, self).__init__(*args, **kwargs)
        self.direction = None
        self._thresh = 0.8 * 0.5 * np.exp(self._log_const)

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

            if self.hypothesis_test() is not None:
                self.status = 'done'

            return

        R = self.model['R'].value
        if self.direction is None:
            step = self._random_step()
            self.direction = np.sign(step)
            self.model['R'].value = step

            if self.hypothesis_test() is not None:
                self.status = 'done'
                return

        else:
            step = self.direction * np.abs(self._random_step())
            self.model['R'].value = R + step

            if self.hypothesis_test() is not None:
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
        llh0 = self.log_lh_h0 + np.log(self.opts['prior'])
        llh1 = self.log_lh_h1 + np.log(1 - self.opts['prior'])

        if llh0 > llh1 and llh0 > np.log(self._thresh):
            return 0
        elif llh1 > llh0 and llh1 > np.log(self._thresh):
            return 1
        else:
            return None

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
