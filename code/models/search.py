import numpy as np


def hill_climbing(model):

    if model.num_samples_left == 0:
        model.debug("Exhausted all samples", level=2)
        return False

    rcurr, scurr = model.curr_val
    model.debug("Current value: %f" % scurr, level=2)

    # this gets the next rotation and similarity, continuing in the
    # direction that we had previously been rotating
    rnext, snext = model.next_val()
    model.debug("Rotate to %d degrees" % np.degrees(rnext), level=2)

    thresh = model.opt.get('h', np.sqrt(model.opt['scale'] / 100.)) ** 2

    # if the next value is less than the current value, try turning
    # around
    if scurr > thresh and snext < scurr:
        model.sample(rcurr)
        model.debug("Rotate to %d degrees" % np.degrees(rcurr), level=2)
    elif scurr <= thresh:
        model.debug("Below threshold, ignoring maxima/minima!", level=2)

    return True
