import matplotlib.pyplot as plt
import numpy as np

from util import draw_stimulus


def stimuli_shapes(**kwargs):
    """Plot the original stimulus and it's rotated counterpart.

    """
    nstim = len(kwargs)

    plt.clf()
    fig = plt.gcf()
    fig.set_figwidth(nstim*2.25)
    fig.set_figheight(2)
    plt.subplots_adjust(top=0.8, bottom=0)

    Xs = sorted(kwargs.keys())
    for i, X in enumerate(Xs):
        plt.subplot(1, nstim, i+1)
        draw_stimulus(kwargs[X])
        plt.title("$%s$" % X)


def stimuli_images(**kwargs):
    nimg = len(kwargs)

    plt.clf()
    fig = plt.gcf()
    fig.set_figwidth(nimg*2.25)
    fig.set_figheight(2)
    plt.subplots_adjust(top=0.8, bottom=0)

    Is = sorted(kwargs.keys())
    for i, I in enumerate(Is):
        plt.subplot(1, nimg, i+1)
        plt.imshow(kwargs[I], cmap='gray', vmin=0, vmax=1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.box('off')
        plt.title("$%s$" % I)


def log_likelihood(R, log_Sr):
    plt.plot(R, log_Sr)

    plt.xlim(0, 2*np.pi)
    plt.xticks(
        [0, np.pi/2., np.pi, 3*np.pi/2., 2*np.pi],
        ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])

    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Log similarity ($\log S(I_b,I_R)$)")
    plt.title("Log likelihood function")


def gp_regression(x, y, xi, yi, xo, yo_mean, yo_var):
    """Plot the original function and the regression estimate.

    """

    plt.plot(x, y, 'r--', label="actual")
    plt.plot(xi, yi, 'ro', label="samples")

    if yo_var is not None:
        lower = yo_mean - np.sqrt(yo_var)
        upper = yo_mean + np.sqrt(yo_var)
        plt.fill_between(xo, lower, upper, color='k', alpha=0.3)

    plt.plot(xo, yo_mean, 'k-', label="estimate")

    plt.xlim(0, 2 * np.pi)
    plt.xticks(
        [0, np.pi / 2., np.pi, 3 * np.pi / 2., 2 * np.pi],
        ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])


def likelihood_modeling_steps(lhr):
    fig = plt.figure()
    plt.clf()

    # overall figure settings
    fig.set_figwidth(15)
    fig.set_figheight(4)
    plt.subplots_adjust(wspace=0.4)

    # plot the regression for S
    plt.subplot(1, 3, 1)
    gp_regression(
        lhr.x, lhr.y, lhr.xi, lhr.yi,
        lhr.x, lhr.mu_S, np.diag(lhr.cov_S))
    plt.title("GPR for $S$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel("Similarity ($S$)")
    plt.legend(loc=0, fontsize=12)
    plt.ylim(1, 2.5)

    # plot the regression for log S
    plt.subplot(1, 3, 2)
    gp_regression(
        lhr.x, np.log(lhr.y), lhr.xi, np.log(lhr.yi),
        lhr.x, lhr.mu_logS, np.diag(lhr.cov_logS))
    plt.title(r"GPR for $\log S$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Log similarity ($\log S$)")
    plt.ylim(np.log(1), np.log(2.5))

    # plot the regression for mu_logS - log_muS
    plt.subplot(1, 3, 3)
    gp_regression(
        lhr.x, lhr.delta, lhr.xc, lhr.yc,
        lhr.x, lhr.mu_Dc, np.diag(lhr.cov_Dc))
    plt.title(r"GPR for $\Delta$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Difference ($\Delta$)")


def likelihood_modeling(lhr):
    plt.figure()
    plt.clf()

    # combine the two regression means to estimate E[Z]
    gp_regression(lhr.x, lhr.y, lhr.xi, lhr.yi, lhr.x, lhr.mean, None)
    plt.title(r"Final Gaussian process regression for $S$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel("Similarity ($S$)")
    plt.legend(loc=0)
    plt.ylim(1, 2.5)
