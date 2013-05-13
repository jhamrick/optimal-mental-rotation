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


def regression(x, y, xi, yi, xo, yo_mean, yo_var, **kwargs):
    """Plot the original function and the regression estimate.

    """

    opt = {
        'title': None,
        'xlabel': "Rotation ($R$)",
        'ylabel': "Similarity ($S$)",
        'legend': True,
    }
    opt.update(kwargs)

    # overall figure settings
    sg.set_figsize(4.5, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, bottom=0.05)

    if x is not None:
        plt.plot(x, y, 'k-', label="actual", linewidth=2)
    if xi is not None:
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

    if xo is not None:
        plt.plot(xo, yo_mean, 'r-', label="estimate", linewidth=2)

    # customize x-axis
    plt.xlim(0, 2 * np.pi)
    plt.xticks(
        [0, np.pi / 2., np.pi, 3 * np.pi / 2., 2 * np.pi],
        ["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"])

    # axis styling
    sg.outward_ticks()
    sg.clear_right()
    sg.clear_top()
    sg.set_scientific(-3, 4, axis='y')

    # title and axis labels
    if opt['title']:
        plt.title(opt['title'])
    if opt['xlabel']:
        plt.xlabel(opt['xlabel'])
    if opt['ylabel']:
        plt.ylabel(opt['ylabel'])

    if opt['legend']:
        plt.legend(loc=0, fontsize=14, frameon=False)


def bq_regression(model):

    # plot the regression for S
    ax_S = plt.subplot(2, 2, 1)
    regression(
        model.R, model.S, model.Ri, model.Si,
        model.R, model.mu_S, np.diag(model.cov_S),
        title="GPR for $S$",
        xlabel=None,
        legend=False)
    sg.no_xticklabels()

    # plot the regression for log S
    ax_logS = plt.subplot(2, 2, 3)
    regression(
        model.R, np.log(model.S + 1), model.Ri, np.log(model.Si + 1),
        model.R, model.mu_logS, np.diag(model.cov_logS),
        title=r"GPR for $\log(S+1)$",
        ylabel=r"Similarity ($\log(S+1)$)",
        legend=False)

    # plot the regression for mu_logS - log_muS
    ax_Dc = plt.subplot(2, 2, 4)
    regression(
        model.R, model.delta, model.Rc, model.Dc,
        model.R, model.mu_Dc, None,
        title=r"GPR for $\Delta_c$",
        ylabel=r"Difference ($\Delta_c$)")
    yt, ytl = plt.yticks()

    # combine the two regression means
    ax_final = plt.subplot(2, 2, 2)
    regression(
        model.R, model.S, model.Ri, model.Si,
        model.R, model.S_mean, model.S_var,
        title=r"Final GPR for $S$",
        xlabel=None,
        ylabel=None,
        legend=False)
    sg.no_xticklabels()

    # align y-axis labels
    sg.align_ylabels(-0.15, ax_S, ax_logS, ax_Dc)
    # sync y-axis limits
    sg.sync_ylims(ax_S, ax_final)

    # overall figure settings
    sg.set_figsize(9, 4)
    plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.05, bottom=0.05)


def vm_regression(model):
    regression(
        model.R, model.S, model.Ri, model.Si,
        model.R, model.S_mean, model.S_var,
        title="Von Mises regression for $S$")


def li_regression(model):
    regression(
        model.R, model.S, model.Ri, model.Si,
        model.R, model.S_mean, model.S_var,
        title="Linear interpolation for $S$")


def likelihood(model):
    regression(
        model.R, model.S, None, None, None, None, None,
        title="Likelihood function",
        legend=False)


def model_rotations():
    fig, axes = plt.subplots(1, 2, sharey=True)
    ax0, ax1 = axes

    for ax in axes:
        sg.outward_ticks(ax=ax)
        sg.clear_right(ax=ax)
        sg.clear_top(ax=ax)

    ax0.set_xlabel("Number of stimuli")
    ax0.set_ylabel("Percent rotated")
    ax0.set_title(r"$h_0$ rotations")

    ax1.set_xlabel(r"True rotation ($R$)")
    ax1.set_title("$h_1$ rotations")
    ax1.set_xticks([0, np.pi / 4., np.pi / 2., 3 * np.pi / 4., np.pi])
    ax1.set_xticklabels([
        "0",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{3\pi}{4}$",
        "$\pi$"
    ])
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0, 105)
    sg.align_xlabels(-0.2, ax0, ax1)

    sg.set_figsize(8, 2.5)
    plt.suptitle("Model Rotations", fontsize=14)
    plt.subplots_adjust(wspace=0.1, top=0.8)

    return fig, axes


def model_z_accuracy(models):
    fig, axes = plt.subplots(1, len(models), sharex=True, sharey=True)
    ax0 = axes[0]

    ax0.set_ylabel(r"Estimated $Z$")
    ax0.set_ylim(0, 1)
    ticks = [0.05, 0.15, 0.25]
    ticklabels = ["%.2f" % x for x in ticks]
    ax0.set_xlim(0, 0.3)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    for i in xrange(len(axes)):
        ax = axes[i]
        model = models[i]
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("True $Z$")

        sg.outward_ticks(ax=ax)
        sg.clear_right(ax=ax)
        sg.clear_top(ax=ax)

    sg.set_figsize(8, 2.5)
    plt.suptitle("Accuracy of $Z$ estimates", fontsize=14)
    plt.subplots_adjust(wspace=0.1, top=0.8)

    return fig, axes
