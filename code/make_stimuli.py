#!/usr/local/bin/python

import numpy as np
import os
import matplotlib
matplotlib.use('AGG')

import util
import model

stim_dir = "../stimuli/"


def make_stimulus(stimnum, npoints, sigma, blur, R, rso):

    # randomly generate the angle of rotation
    theta = rso.uniform(0, 2*np.pi)

    # generate the original shapes
    Xa = util.make_stimulus(npoints, rso)

    # not the same
    Xb0 = util.rotate(util.reflect(Xa), theta)
    # the same
    Xb1 = util.rotate(Xa, theta)

    # render images
    Ia = util.blur(util.make_image(Xa, sigma=sigma, rso=rso), blur)
    Ib0 = util.blur(util.make_image(Xb0, sigma=sigma, rso=rso), blur)
    Ib1 = util.blur(util.make_image(Xb1, sigma=sigma, rso=rso), blur)

    # the angular increment between each rotation
    r = R[1] - R[0]

    for Xb, Ib, hyp in [(Xb0, Ib0, 'h0'), (Xb1, Ib1, 'h1')]:
        stimname = "%s_%s" % (hyp, stimnum)

        # array to store all the rotated shapes
        Xm = np.zeros(R.shape + Xa.shape)
        Xm[0] = Xa.copy()
        # array to store all the rotated images
        Im = np.zeros(R.shape + Ia.shape)
        Im[0] = Ia.copy()
        # array to store all the similarities
        Sr = np.zeros(R.shape)
        Sr[0] = model.similarity(Ib, Im[0], sf=blur)

        # create stimuli directory, if it does not exist
        if not os.path.exists(stim_dir):
            os.makedirs(stim_dir)

        # skip the computationally intensive part, if it exists
        path = os.path.join(stim_dir, stimname + ".npz")
        if os.path.exists(path):
            print "Exists: %s" % path
            return

        # generate mental images and then compare them
        for i in xrange(1, R.size):
            Xm[i] = util.rotate(Xm[i-1], r)
            Im[i] = util.blur(util.render(Xm[i]), blur)
            Sr[i] = model.similarity(Ib, Im[i], sf=blur)
            print "[%d / %d] %f" % (i+1, R.size, Sr[i])

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
            # images
            Im=Im,
            Ib=Ib,
            # similarity
            Sr=Sr,
        )

if __name__ == "__main__":
    # random state, for reproducibility
    rso = np.random.RandomState(0)

    # config variables
    npoints = 5
    sigma = 0.0
    blur = 5.8
    nstim = 10

    # all the angles we want to try
    R = np.linspace(0, 2*np.pi, 360)

    for i in xrange(nstim):
        stimnum = "%03d" % i
        make_stimulus(stimnum, npoints, sigma, blur, R, rso)
