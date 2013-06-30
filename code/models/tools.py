import os
import numpy as np


def print_line(char='-', verbose=1):
    if verbose > 0:
        print "\n" + char*70


def run_model(stim, model, opt):

    modelname = model.__name__
    name = "%s-%s.npz" % (modelname, stim)
    datadir = opt['data_dir']
    path = os.path.abspath(os.path.join(datadir, name))
    lockfile = path + ".lock"

    # skip this simulation, if it already exists
    if os.path.exists(path) or os.path.exists(lockfile):
        print_line(char='#')
        print "'%s' exists, skipping" % path
        return

    print_line(char='#')
    print "%s (%s)" % (stim, path)

    # make the data directories if they don't exist
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    # create a lockfile
    with open(lockfile, 'w') as fh:
        fh.write("%s\n" % path)

    # number of samples
    nsamp = opt['nsamps']

    # how many points were sampled
    samps = np.zeros(nsamp)
    # the estimate of Z
    Z = np.empty((nsamp, 3))
    # the likelihood ratio
    ratio = np.empty((nsamp, 3))
    # which hypothesis was accepted
    hyp = np.empty(nsamp)

    # load the stimulus
    theta, Xa, Xb, Xm, Ia, Ib, Im, R = load_stimulus(stim, opt['stim_dir'])

    for i in xrange(nsamp):
        # run the model
        m = model(Ia[i], Ib[i], Im[:, i], R, **opt)
        m.run()

        # fill in the data arrays
        samps[i] = len(m.ix) / float(m._rotations.size)
        Z[i, 0] = m.Z_mean
        Z[i, 1:] = m.Z_var
        ratio[i] = m.likelihood_ratio()
        hyp[i] = m.ratio_test(level=10)

    if os.path.exists(lockfile):
        np.savez(
            path,
            samps=samps,
            Z=Z,
            ratio=ratio,
            hyp=hyp
        )

        try:
            # remove lockfile
            os.remove(lockfile)
        except OSError:
            pass


def run_all(stims, model, opt):
    # get stimuli -- if none passed, then run all of them
    if len(stims) == 0:
        stims = find_stims(opt['stim_dir'])
    # run each stim
    for stim in stims:
        run_model(stim, model, opt)


def load_stimulus(stimname, stim_dir):
    stimf = np.load(os.path.join(stim_dir, stimname + ".npz"))

    # rotation
    R = stimf['R']
    theta = stimf['theta']
    # shapes
    Xm = stimf['Xm']
    Xa = Xm[0]
    Xb = stimf['Xb']
    # observations
    Im = stimf['Im']
    Ia = Im[0]
    Ib = stimf['Ib']

    stimf.close()
    return theta, Xa, Xb, Xm, Ia, Ib, Im, R


def find_stims(stim_dir):
    stims = sorted([os.path.splitext(x)[0] for x in os.listdir(stim_dir)])
    return stims
