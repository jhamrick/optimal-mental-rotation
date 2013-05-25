import numpy as np


def hill_climbing(model):

    inext = model._icurr + 1
    iprev = model._icurr - 1

    n = model._rotations.size
    if inext >= n or np.abs(iprev) >= n:
        model.debug("Exhausted all samples", level=2)
        return None

    rcurr = model._rotations[model._icurr]
    rnext = model._rotations[inext]
    rprev = model._rotations[iprev]

    scurr = model.sample(rcurr)
    snext = model.sample(rnext)
    sprev = model.sample(rprev)

    model.debug("Current value: %f" % scurr, level=2)

    choose_next = False
    choose_prev = False

    # we're at a super small scale, so we want to ignore
    # minima/maxima here
    if scurr <= (model.opt['scale'] / 10.):
        model.debug("Below threshold, ignoring maxima/minima!", level=2)
        if (snext > sprev) and (inext != model._ilast):
            choose_next = True
        elif (sprev > snext) and (iprev != model._ilast):
            choose_prev = True
        elif (model._icurr - model._ilast) > 0:
            choose_next = True
        else:
            choose_prev = True
    elif (snext > scurr) and (snext > sprev):
        choose_next = True
    elif (sprev > scurr) and (sprev > snext):
        choose_prev = True

    if choose_next and not choose_prev:
        model.debug("Choosing next: %d" % inext, level=2)
        icurr = inext
    elif choose_prev and not choose_next:
        model.debug("Choosing prev: %d" % iprev, level=2)
        icurr = iprev
    else:
        model.debug("Done", level=2)
        icurr = None

    return icurr
