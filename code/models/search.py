import numpy as np


def hill_climbing(model):

    if model.num_samples_left == 0:
        model.debug("Exhausted all samples", level=2)
        return False

    rcurr, scurr = model.curr_val
    d = int(np.round(np.degrees(rcurr)))
    model.debug("Current: S(%d) = %f" % (d, scurr), level=1)

    # this gets the next rotation and similarity, continuing in the
    # direction that we had previously been rotating
    rnext, snext = model.next_val()

    thresh = model.opt.get('h', None)
    if thresh is None:
        thresh = np.sqrt(model.opt['scale'] / 100.)
    thresh = thresh ** 2

    if scurr > thresh and snext < scurr:
        # if the next value is less than the current value, try
        # turning around -- first go back to the current sample
        model.sample(rcurr)
        # then go one further
        rprev, sprev = model.next_val()
        # if this sample is still less than current, then we've found
        # a maximum
        if sprev < scurr:
            return False

    elif scurr <= thresh:
        model.debug("Below threshold, ignoring maxima/minima!", level=2)

    return True
