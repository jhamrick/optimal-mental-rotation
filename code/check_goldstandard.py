"""Check the "gold standard" model to make sure it picks the right
hypothesis.

"""

import os
import numpy as np
import model
import util

opt = {
    # print extra information
    'verbose': False,
    # data scale, to avoid underflow
    'scale': 1,
    # the amount we rotate by
    'dr': 20,
    # standard deviation in similarity function
    'sigma': 0.2,
    # prior over angles
    'prior_R': 1. / (2 * np.pi),
}

# array to hold number of wrong stimuli
num_wrong = np.zeros(2)
num_total = np.zeros(2)

stims = sorted([os.path.splitext(x)[0] for x in os.listdir('../stimuli')])
for stim in stims:
    print "\n" + "-"*70
    print stim

    # get the true hypothesis
    true_hyp = int(stim[1])
    num_total[true_hyp] += 1

    # load the stimulus
    theta, Xa, Xb, Xm, R = util.load_stimulus(stim)

    # run the naive model
    gs = model.GoldStandard(Xa, Xb, Xm, R, **opt)
    gs.run()

    # compute the ratio
    hyp = gs.ratio_test(level=10)

    if hyp != int(stim[1]):
        num_wrong[true_hyp] += 1
        print "!!!!! Wrong hypothesis chosen"

print "\n" + "#"*70
print "Done. Fraction of incorrect hypotheses:"
print num_wrong.astype('f8') / num_total
