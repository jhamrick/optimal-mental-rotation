import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as nd
import tempfile

from joblib import Memory
memory = Memory(cachedir='cache', verbose=0)

from snippets.graphing import save


def make_stimulus(npoints, rso):
    """Make a shape with `npoints` vertices."""
    X = rso.rand(npoints, 2)
    X = X - np.mean(X, axis=0)
    r = np.arctan2(X[:, 1], X[:, 0])
    idx = np.argsort(r)
    X = np.concatenate([X[idx], X[[idx[0]]]], axis=0)
    return X


def make_image(X, sigma, rso):
    """Jitter the vertices of the shape using Gaussian noise with variance
    `sigma`.

    """
    if sigma == 0:
        I = X.copy()
    else:
        I = X + rso.normal(0, sigma, X.shape)
    return I


@memory.cache
def render(I):
    """Render the shape into pixel-space.

    Note: this function is memoized. It is also horribly inefficient.

    """
    # create a temporary file to render the shape to
    fh = tempfile.NamedTemporaryFile()
    filename = fh.name + '.png'
    fh.close()

    # plot the shape
    fig = plt.figure()
    fig.clf()
    fig.set_figwidth(2)
    fig.set_figheight(2)
    plt.plot(I[:, 0], I[:, 1],
             color='k',
             linewidth=2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.box('off')
    # save it to the temporary file name
    save(filename, width=4, height=4)

    # load it back in as pixels
    img = mpimg.imread(filename)
    # convert it to grayscale to save space
    grayscale = np.sqrt(np.sum(img[:, :, :-1]**2, axis=2) / 3.)

    return grayscale


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


def plot_stimuli(Xa, Ia, Xb, Ib):
    """Plot the original stimulus and it's rotated counterpart, as well as
    noisy versions of each.

    """
    plt.clf()
    fig = plt.gcf()
    fig.set_figwidth(9)
    fig.set_figheight(4)

    plt.subplot(1, 2, 1)
    plt.plot(Xa[:, 0], Xa[:, 1],
             color='k',
             linewidth=2,
             label="$X_a$")
    plt.plot(Ia[:, 0], Ia[:, 1],
             color='k',
             alpha=0.5,
             linewidth=2,
             label="$I_a$")
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(loc=4, numpoints=1)

    plt.subplot(1, 2, 2)
    plt.plot(Xb[:, 0], Xb[:, 1],
             color='b',
             linewidth=2,
             label="$X_b$")
    plt.plot(Ib[:, 0], Ib[:, 1],
             color='b',
             alpha=0.5,
             linewidth=2,
             label="$I_b$")
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(loc=4, numpoints=1)
