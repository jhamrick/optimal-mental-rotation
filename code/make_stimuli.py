#!/usr/local/bin/python

import numpy as np
import os
import matplotlib
matplotlib.use('AGG')

import util

stim_dir = util.STIM_DIR


def make_stimulus(stimnum, npoints, nsamps, sigma, R, rso):

    # generate the original shapes
    Xa = util.make_stimulus(npoints, rso)

    for theta_deg in xrange(0, 360, 20):
        theta = np.radians(theta_deg)

        # not the same
        Xb0 = util.rotate(util.reflect(Xa), theta)
        # the same
        Xb1 = util.rotate(Xa, theta)

        # the angular increment between each rotation
        r = R[1] - R[0]

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
                    # print "[%d / %d]" % (i+1, R.size)

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
        make_stimulus(stimnum, npoints, nsamps, sigma, R, rso)
