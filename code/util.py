import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL
import models
import os
import yaml

from snippets.graphing import plot_to_array

OPT_PATH = "options.yml"


def load_opt():
    with open(OPT_PATH, 'r') as fh:
        opt = yaml.load(fh)
    return opt


def make_stimulus(npoints, rso):
    """Make a shape with `npoints` vertices."""
    if hasattr(npoints, '__iter__'):
        npoints = rso.randint(npoints[0], npoints[1]+1)
    # pick random points
    X_ = rso.rand(npoints, 2)
    # normalize points
    X = X_ - np.mean(X_, axis=0)
    # normalize the shape's size, so the furthest point is distance 1
    # away from the origin
    X = 0.9 * X / np.max(np.sqrt(np.sum(X ** 2, axis=1)))
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


def compress_sims(name, data_dir):
    path = os.path.join(data_dir, name)
    print "Compressing '%s'..." % path
    sims = sorted(os.listdir(path))
    nsim = len(sims)

    # arrays to hold data
    stims = []
    samps = None
    Z = None
    ratio = None
    hyp = None

    for sidx, sim in enumerate(sims):
        data = np.load(os.path.join(path, sim))

        # create arrays
        if sidx == 0:
            samps = np.empty((nsim,) + data['samps'].shape)
            Z = np.empty((nsim,) + data['Z'].shape)
            ratio = np.empty((nsim,) + data['ratio'].shape)
            hyp = np.empty((nsim,) + data['hyp'].shape)

        # stims
        stim = os.path.splitext(sim)[0].split("-")[1]
        stims.append(stim)
        # samps
        samps[sidx] = data['samps']
        # Z
        Z[sidx] = data['Z']
        # ratio
        ratio[sidx] = data['ratio']
        # hyp
        hyp[sidx] = data['hyp']
        data.close()

    newpath = os.path.join(data_dir, name + ".npz")
    np.savez(
        newpath,
        stims=stims,
        samps=samps,
        Z=Z,
        ratio=ratio,
        hyp=hyp)
    print "-> Saved to '%s'" % newpath


def load_sims(name, data_dir):
    path = os.path.join(data_dir, name + '.npz')
    data = np.load(path)
    stims = data['stims']
    samps = data['samps']
    Z = data['Z']
    ratio = data['ratio']
    hyp = data['hyp']
    data.close()
    return stims, samps, Z, ratio, hyp


def rand_params(*args):
    params = []
    for param in args:
        if param == 'h':
            params.append(np.random.uniform(0, 2))
        elif param == 'w':
            params.append(np.random.uniform(np.pi / 32., np.pi / 2.))
        elif param == 'p':
            params.append(np.random.uniform(0.33, 3))
        elif param == 's':
            params.append(np.random.uniform(0, 0.5))
    return tuple(params)

run_model = models.tools.run_model
run_all = models.tools.run_all
load_stimulus = models.tools.load_stimulus
find_stims = models.tools.find_stims
