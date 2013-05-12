def hill_climbing(model):

    inext = model._icurr + 1
    iprev = model._icurr - 1

    rcurr = model._rotations[model._icurr]
    rnext = model._rotations[inext]
    rprev = model._rotations[iprev]

    scurr = model.sample(rcurr)
    snext = model.sample(rnext)
    sprev = model.sample(rprev)

    if model.opt['verbose']:
        print "Current value: %f" % scurr

    choose_next = False
    choose_prev = False

    # we're at a super small scale, so we want to ignore
    # minima/maxima here
    if scurr <= (model.opt['scale'] / 1000.):
        if model.opt['verbose']:
            print "Below threshold, ignoring maxima/minima!"
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
        if model.opt['verbose']:
            print "Choosing next: %d" % inext
        icurr = inext
    elif choose_prev and not choose_next:
        if model.opt['verbose']:
            print "Choosing prev: %d" % iprev
        icurr = iprev
    else:
        if model.opt['verbose']:
            print "Done"
        icurr = None
        raise StopIteration

    return icurr
