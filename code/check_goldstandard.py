"""Check the "gold standard" model to make sure it picks the right
hypothesis.

"""

import os
import numpy as np
import model
import util

# data scale, to avoid underflow
SCALE = 50
# sigma, used in similarity function
SIGMA = 0.2
# prior over angles
pR = 1 / 360.
# the amount we rotate by
dr = 20
# print extra information
verbose = False

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

    # scale the similarities, so we don't get underflow
    Sr_scale = (Xa.shape[0]-1) / (2 * np.pi * SIGMA) * SCALE
    Sr = np.array([model.similarity(Xb, X, sf=SIGMA) for X in Xm]) * Sr_scale

    # compute the joint probability of hypothesis 0
    p_Xa = model.log_prior_X(Xa)
    p_Xb = model.log_prior_X(Xb)
    p_XaXb_h0 = p_Xa + p_Xb

    # run the naive model
    gs = model.GoldStandard(R, Sr, dr, pR, verbose=verbose)
    gs.run()

    # compute the ratio
    ratio_naive = model.likelihood_ratio(gs, Sr_scale, p_Xa, p_XaXb_h0)
    hyp = model.ratio_test(ratio_naive)

    if hyp != int(stim[1]):
        num_wrong[true_hyp] += 1
        print "!!!!! Wrong hypothesis chosen"

print "\n" + "#"*70
print "Done. Fraction of incorrect hypotheses:"
print num_wrong.astype('f8') / num_total
