import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL

from snippets.graphing import plot_to_array


def make_stimulus(npoints, rso):
    """Make a shape with `npoints` vertices."""
    X = rso.rand(npoints, 2)
    X = X - np.mean(X, axis=0)
    r = np.arctan2(X[:, 1], X[:, 0])
    idx = np.argsort(r)
    X = np.concatenate([X[idx], X[[idx[0]]]], axis=0)
    return X


def draw_stimulus(X):
    plt.plot(X[:, 0], X[:, 1],
             color='k',
             linewidth=2)
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


def make_image(X, sigma, rso):
    """Jitter the vertices of the shape using Gaussian noise with variance
    `sigma`.

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
