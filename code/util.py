import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL
import os
import yaml

from snippets.graphing import plot_to_array
from snippets.safemath import MIN_LOG, MAX_LOG
from snippets.safemath import log_clip, safe_multiply, safe_log

STIM_DIR = "../stimuli"
DATA_DIR = "../data"


def make_stimulus(npoints, rso):
    """Make a shape with `npoints` vertices."""
    # pick random points
    X = rso.rand(npoints, 2)
    # subtract off the mean
    X = X - np.mean(X, axis=0)
    # normalize the shape's size, so the furthest point is distance 1
    # away from the origin
    X = X / np.max(np.sqrt(np.sum(X ** 2, axis=1)))
    # order them by angle, so they plot nicely
    r = np.arctan2(X[:, 1], X[:, 0])
    idx = np.argsort(r)
    X = np.concatenate([X[idx], X[[idx[0]]]], axis=0)
    return X


def draw_stimulus(X, **kwargs):
    plt.plot(X[:, 0], X[:, 1],
             color='k',
             linewidth=2,
             **kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis([-1, 1, -1, 1])
    plt.box('off')


def render(X):
    """Render the shape into pixel-space."""

    # plot the stimulus
    fig = plt.figure()
    fig.clf()
    fig.set_figwidth(2)
    fig.set_figheight(2)
    # turn off the gray frame
    fig.frameon = False
    # make the plot fit the full figure width, since we don't have
    # axis/tick labels or a title, etc.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    draw_stimulus(X)

    # convert the image to pixels and close it
    img = plot_to_array(fig)
    plt.close(fig)

    # convert it to grayscale to save space
    grayscale = np.sqrt(np.sum(img**2, axis=2) / 3.)

    return grayscale


def observe(X, sigma, rso):
    """Jitter the vertices of the shape using Gaussian noise with variance
    `sigma`.

    """
    if sigma == 0:
        I = X.copy()
    else:
        I = X + rso.normal(0, sigma, X.shape)
    return I


def make_image(X, sigma, rso):
    """Jitter the vertices of the shape using Gaussian noise with variance
    `sigma` and render the resulting shape.

    """
    if sigma == 0:
        I = X.copy()
    else:
        I = X + rso.normal(0, sigma, X.shape)
    rI = render(I)
    return rI


def blur(img, blur):
    """Blur a 2d image with Gaussian variance equal to `blur`."""
    blurred = nd.gaussian_filter(img, blur)
    return blurred


def make_rotation_matrix(theta):
    """Create a 2-dimensional rotation matrix from the angle `theta`, in
    radians.

    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R


def rotate(Xa, theta):
    """Rotate a set of points given in `Xa` (shape Nx2) by `theta` radians.

    """
    R = make_rotation_matrix(theta)
    Xb = np.dot(Xa, R.T)
    return Xb


def rotate_image(I, theta):
    img = PIL.Image.fromarray(1-I)
    rimg = img.rotate(np.degrees(theta))
    rI = 1-np.array(rimg)
    return rI


def reflect(Xa):
    """Reflect a set of points given in `Xa` (shape Nx2) about the y-axis.

    """
    M = np.array([
        [-1, 0],
        [0, 1]
    ])
    Xb = np.dot(Xa, M)
    return Xb


def reflect_image(I):
    img = PIL.Image.fromarray(I)
    rimg = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    rI = np.array(rimg)
    return rI


def load_stimulus(stimname):
    stimf = np.load(os.path.join(STIM_DIR, stimname + ".npz"))

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


def print_line(char='-', verbose=True):
    if verbose:
        print "\n" + char*70


def run_model(stims, model, opt):

    # number of stims
    nstim = opt['nstim']
    # number of samples
    nsamp = opt['nsamps']

    # how many points were sampled
    samps = np.zeros((nstim, nsamp))
    # the estimate of Z
    Z = np.empty((nstim, nsamp, 2))
    # the likelihood ratio
    ratio = np.empty((nstim, nsamp, 3))
    # which hypothesis was accepted
    hyp = np.empty((nstim, nsamp))

    for sidx, stim in enumerate(stims):
        print_line(char='#')
        print stim

        # load the stimulus
        theta, Xa, Xb, Xm, Ia, Ib, Im, R = load_stimulus(stim)

        for i in xrange(nsamp):
            # run the model
            m = model(Ia[i], Ib[i], Im[:, i], R, **opt)
            m.run()

            # fill in the data arrays
            samps[sidx, i] = len(m.ix) / float(m._rotations.size)
            Z[sidx, i] = (m.Z_mean, m.Z_var)
            ratio[sidx, i] = m.likelihood_ratio()
            hyp[sidx, i] = m.ratio_test(level=10)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    name = type(m).__name__
    if name.endswith("Model"):
        name = name[:-len("Model")]
    path = os.path.join(DATA_DIR, name)
    np.savez(
        path,
        stims=stims,
        samps=samps,
        Z=Z,
        ratio=ratio,
        hyp=hyp
    )


def load_opt():
    with open('options.yml', 'r') as fh:
        opt = yaml.load(fh)
    return opt


def find_stims():
    stims = sorted([os.path.splitext(x)[0] for x in os.listdir(STIM_DIR)])
    return stims


def load_sims(name):
    path = os.path.join(DATA_DIR, name + '.npz')
    data = np.load(path)
    stims = data['stims']
    samps = data['samps']
    Z = data['Z']
    ratio = data['ratio']
    hyp = data['hyp']
    data.close()
    return stims, samps, Z, ratio, hyp
