#!/usr/local/bin/python

import numpy as np
import os
import matplotlib
matplotlib.use('AGG')

import util
from model_gs import GoldStandardModel

stim_dir = util.STIM_DIR


def check_discrim(Xa, R, r, opt):
    # not the same
    Xb0 = util.rotate(util.reflect(Xa), 0)
    Xb1 = util.rotate(Xa, 0)

    # generate mental images and then compare them
    Xm = np.zeros(R.shape + Xa.shape)
    Xm[0] = Xa.copy()
    for i in xrange(1, R.size):
        Xm[i] = util.rotate(Xm[i-1], r)

    gs0 = GoldStandardModel(Xa, Xb0, Xm, R, **opt)
    gs0.run()
    r0 = gs0.likelihood_ratio()[0]

    gs1 = GoldStandardModel(Xa, Xb1, Xm, R, **opt)
    gs1.run()
    r1 = gs1.likelihood_ratio()[0]

    thresh = 1.5
    good0 = r0 < (1 / thresh)
    good1 = r1 > thresh

    if good0 or good1:
        print r0, r1
    return good0 and good1


def make_stimulus(stimnum, npoints, nsamps, sigma, R, rso, opt):

    # the angular increment between each rotation
    r = R[1] - R[0]

    # generate the original shapes
    while True:
        Xa = util.make_stimulus(npoints, rso)
        discrim = check_discrim(Xa, R, r, opt)
        if discrim:
            break

    for theta_deg in xrange(0, 360, 20):
        theta = np.radians(theta_deg)

        # not the same
        Xb0 = util.rotate(util.reflect(Xa), theta)
        # the same
        Xb1 = util.rotate(Xa, theta)

        for Xb, hyp in [(Xb0, 'h0'), (Xb1, 'h1')]:
            stimname = "%s_%03d_%s" % (stimnum, theta_deg, hyp)

            # create stimuli directory, if it does not exist
            if not os.path.exists(stim_dir):
                os.makedirs(stim_dir)

            # skip the computationally intensive part, if it exists
            path = os.path.join(stim_dir, stimname + ".npz")
            if os.path.exists(path):
                print "Exists: %s" % path
                return

            # array to store the observed shapes
            Ib = np.zeros((nsamps,) + Xa.shape)
            Im = np.zeros(R.shape + (nsamps,) + Xa.shape)

            for sampnum in xrange(nsamps):
                Ia = util.observe(Xa, sigma, rso)
                Ib[sampnum] = util.observe(Xb, sigma, rso)

                # array to store all the rotated shapes
                Xm = np.zeros(R.shape + Xa.shape)
                Xm[0] = Xa.copy()
                Im[0, sampnum] = Ia.copy()

                # generate mental images and then compare them
                for i in xrange(1, R.size):
                    Xm[i] = util.rotate(Xm[i-1], r)
                    Im[i] = util.rotate(Im[i-1], r)

            # save data to numpy arrays
            print "Saving: %s" % path
            np.savez(
                path,
                # rotations
                R=R,
                theta=theta,
                # shapes
                Xm=Xm,
                Xb=Xb,
                # observations
                Im=Im,
                Ib=Ib
            )

if __name__ == "__main__":
    # config variables
    opt = util.load_opt()
    opt['verbose'] = -1
    npoints = opt['npoints']
    nstim = opt['nstim']
    nsamps = opt['nsamps']
    sigma = opt['sigma_p']

    # random state, for reproducibility
    rso = np.random.RandomState(0)

    # all the angles we want to try
    R = np.linspace(0, 2*np.pi, 361)

    for i in xrange(nstim):
        stimnum = "%03d" % i
        make_stimulus(stimnum, npoints, nsamps, sigma, R, rso, opt)
