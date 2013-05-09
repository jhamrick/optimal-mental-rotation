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

    ax = plt.gca()
    ax.tick_params(direction='out')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()


def likelihood_modeling(lhr):
    fig = plt.figure()
    plt.clf()

    labelx = -0.15

    # overall figure settings
    fig.set_figwidth(9)
    fig.set_figheight(5)
    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, bottom=0.05)

    # plot the regression for S
    plt.subplot(2, 2, 1)
    gp_regression(
        lhr.x, lhr.y + 1, lhr.xi, lhr.yi + 1,
        lhr.x, lhr.mu_S + 1, np.diag(lhr.cov_S))
    plt.title("GPR for $S$")
    plt.xticks(plt.xticks()[0], [])
    plt.ylabel("Similarity ($S$)")
    plt.gca().yaxis.set_label_coords(labelx, 0.5)
    ylim1 = plt.ylim()

    # plot the regression for log S
    plt.subplot(2, 2, 3)
    gp_regression(
        lhr.x, np.log(lhr.y + 1), lhr.xi, np.log(lhr.yi + 1),
        lhr.x, lhr.mu_logS, np.diag(lhr.cov_logS))
    plt.title(r"GPR for $\log S$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Similarity ($\log S$)")
    plt.gca().yaxis.set_label_coords(labelx, 0.5)
    ylim2 = np.exp(plt.ylim())

    # plot the regression for mu_logS - log_muS
    plt.subplot(2, 2, 4)
    gp_regression(
        lhr.x, lhr.delta, lhr.xc, lhr.yc,
        lhr.x, lhr.mu_Dc, np.diag(lhr.cov_Dc))
    plt.title(r"GPR for $\Delta_c$")
    plt.xlabel("Rotation ($R$)")
    plt.ylabel(r"Difference ($\Delta_c$)")
    plt.legend(loc=0, fontsize=14, frameon=False)
    yt, ytl = plt.yticks()

    # combine the two regression means to estimate E[Z]
    plt.subplot(2, 2, 2)
    gp_regression(
        lhr.x, lhr.y + 1, lhr.xi, lhr.yi + 1,
        lhr.x, lhr.mean + 1, np.diag(lhr.cov_logS))
    plt.title(r"Final GPR for $S$")
    plt.xticks(plt.xticks()[0], [])
    ylim3 = plt.ylim()

    # figure out appropriate y-axis limits
    ylims = np.array([ylim1, ylim2, ylim3])
    ylo = max(np.min(ylims[:, 0]), 1)
    yhi = np.max(ylims[:, 1])
    yr = (yhi - ylo) / 10

    # set these axis limits
    plt.subplot(2, 2, 1)
    plt.ylim(ylo, yhi)
    plt.subplot(2, 2, 2)
    plt.ylim(ylo, yhi)
    plt.subplot(2, 2, 3)
    plt.ylim(np.log(ylo), np.log(yhi))
    plt.subplot(2, 2, 4)
    yl = plt.ylim()
    plt.ylim(min(yl[0], -yr / 2.), max(yr, yl[1]))
