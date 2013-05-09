import matplotlib.pyplot as plt
import numpy as np

from util import draw_stimulus
import snippets.graphing as sg


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


def likelihood(R, Sr):
    plt.plot(R, Sr)

    plt.xlim(0, 2*np.pi)
    plt.xticks(
        [0, np.pi/2., np.pi, 3*np.pi/2., 2*np.pi],
        ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])

    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Similarity ($S(I_b,I_R)$)")
    plt.title("Likelihood function")


def regression(x, y, xi, yi, xo, yo_mean, yo_var):
    """Plot the original function and the regression estimate.

    """

    plt.plot(x, y, 'k-', label="actual", linewidth=2)
    plt.plot(xi, yi, 'ro', label="samples")

    if yo_var is not None:
        # hack, for if there are zero or negative variances
        yv = np.abs(yo_var)
        ys = np.zeros(yv.shape)
        ys[yv != 0] = np.sqrt(yv[yv != 0])
        # compute upper and lower bounds
        lower = yo_mean - ys
        upper = yo_mean + ys
        plt.fill_between(xo, lower, upper, color='r', alpha=0.25)

    plt.plot(xo, yo_mean, 'r-', label="estimate", linewidth=2)

    plt.xlim(0, 2 * np.pi)
    plt.xticks(
        [0, np.pi / 2., np.pi, 3 * np.pi / 2., 2 * np.pi],
        ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])

    sg.outward_ticks()
    sg.clear_right()
    sg.clear_top()


def bq_likelihood_regression(bq):
    labelx = -0.15

    # overall figure settings
    sg.set_figsize(9, 5)
    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, bottom=0.05)

    # plot the regression for S
    plt.subplot(2, 2, 1)
    regression(
        bq.x, bq.y, bq.xi, bq.yi,
        bq.x, bq.mu_S, np.diag(bq.cov_S))
    plt.title("GPR for $S$")
    plt.ylabel("Similarity ($S$)")
    sg.set_ylabel_coords(labelx)
    sg.no_xticklabels()
    ylim1 = plt.ylim()

    # plot the regression for log S
    plt.subplot(2, 2, 3)
    regression(
        bq.x, np.log(bq.y + 1), bq.xi, np.log(bq.yi + 1),
        bq.x, bq.mu_logS, np.diag(bq.cov_logS))
    plt.title(r"GPR for $\log(S+1)$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Similarity ($\log(S+1)$)")
    sg.set_ylabel_coords(labelx)

    # plot the regression for mu_logS - log_muS
    plt.subplot(2, 2, 4)
    regression(
        bq.x, bq.delta, bq.xc, bq.yc,
        bq.x, bq.mu_Dc, np.diag(bq.cov_Dc))
    plt.title(r"GPR for $\Delta_c$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Difference ($\Delta_c$)")
    plt.legend(loc=0, fontsize=14, frameon=False)
    yt, ytl = plt.yticks()

    # combine the two regression means to estimate E[Z]
    plt.subplot(2, 2, 2)
    regression(
        bq.x, bq.y, bq.xi, bq.yi,
        bq.x, bq.mean, np.diag(bq.cov_logS))
    plt.title(r"Final GPR for $S$")
    sg.no_xticklabels()
    ylim2 = plt.ylim()

    # figure out appropriate y-axis limits
    ylims = np.array([ylim1, ylim2])
    ylo = ylims[:, 0].min()
    yhi = ylims[:, 1].max()

    # set these axis limits
    plt.subplot(2, 2, 1)
    plt.ylim(ylo, yhi)
    plt.subplot(2, 2, 2)
    plt.ylim(ylo, yhi)
    # plt.subplot(2, 2, 3)
    # plt.ylim(np.log(ylo), np.log(yhi))
    # plt.subplot(2, 2, 4)
    # yl = plt.ylim()
    # plt.ylim(min(yl[0], -yr / 2.), max(yr, yl[1]))


def parametric_regression(pr):
    # overall figure settings
    sg.set_figsize(4, 4)

    # plot the regression for S
    regression(
        pr.x, pr.y, pr.xi, pr.yi,
        pr.x, pr.mean, None)
    plt.title("Parametric regression for $S$")
    plt.ylabel("Similarity ($S$)")
