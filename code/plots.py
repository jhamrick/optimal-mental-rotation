import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from util import draw_stimulus
import snippets.graphing as sg


def stimuli_shapes(**kwargs):
    """Plot the original stimulus and it's rotated counterpart.

    """
    nstim = len(kwargs)

    plt.figure()
    sg.set_figsize(nstim*2, 2)
    plt.subplots_adjust(
        top=0.8, bottom=0, left=0, right=1,
        wspace=0)

    Xs = sorted(kwargs.keys())
    for i, X in enumerate(Xs):
        plt.subplot(1, nstim, i+1)
        draw_stimulus(kwargs[X])
        plt.title("$%s$" % X)


def stimuli_samples(**kwargs):
    """Plot the observed stimulus and it's rotated counterpart.

    """
    nstim = len(kwargs)

    plt.figure()
    sg.set_figsize(nstim*2, 2)
    plt.subplots_adjust(
        top=0.8, bottom=0, left=0, right=1,
        wspace=0)

    Xs = sorted(kwargs.keys())
    for i, X in enumerate(Xs):
        plt.subplot(1, nstim, i+1)
        plt.title("$%s$" % X)
        for j in xrange(kwargs[X].shape[0]):
            draw_stimulus(kwargs[X][j], alpha=0.2)


def stimuli_images(**kwargs):
    nimg = len(kwargs)

    plt.figure()
    sg.set_figsize(nimg*2, 2)
    plt.subplots_adjust(
        top=0.8, bottom=0, left=0, right=1,
        wspace=0)

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
    sg.set_figsize(5, 3)
    plt.subplots_adjust(
        wspace=0.2, hspace=0.3,
        left=0.15, bottom=0.2, right=0.95)

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
    sg.set_scientific(-2, 3, axis='y')

    # title and axis labels
    if opt['title']:
        plt.title(opt['title'])
    if opt['xlabel']:
        plt.xlabel(opt['xlabel'])
    if opt['ylabel']:
        plt.ylabel(opt['ylabel'])

    if opt['legend']:
        plt.legend(loc=0, fontsize=12, frameon=False)


def bq_regression(model):

    R = model.R
    S = model.S
    logS = model._bq.log_transform(S)
    Ri = model.Ri
    Si = model.Si
    logSi = model._bq.log_transform(Si)
    delta = model._bq.delta
    Rc = model._bq.Rc
    Dc = model._bq.Dc

    gp_S = model._bq.gp_S
    gp_logS = model._bq.gp_logS
    gp_Dc = model._bq.gp_Dc

    # plot the regression for S
    ax_S = plt.subplot(2, 2, 1)
    regression(
        R, S, Ri, Si,
        R, gp_S.mean(R)[:, 0], np.diag(gp_S.cov(R)),
        title="GPR for $S$",
        xlabel=None,
        legend=False)
    ax_S.legend(
        loc='upper center',
        bbox_to_anchor=(1.08, 1.48),
        frameon=False,
        fontsize=12,
        ncol=3)
    sg.no_xticklabels()

    # plot the regression for log S
    ax_logS = plt.subplot(2, 2, 3)
    regression(
        R, logS, Ri, logSi,
        R, gp_logS.mean(R)[:, 0], np.diag(gp_logS.cov(R)),
        title=r"GPR for $\log(S+1)$",
        ylabel=r"Similarity ($\log(S+1)$)",
        legend=False)

    # plot the regression for mu_logS - log_muS
    ax_Dc = plt.subplot(2, 2, 4)
    regression(
        R, delta, Rc, Dc,
        R, gp_Dc.mean(R)[:, 0], np.diag(gp_Dc.cov(R)),
        title=r"GPR for $\Delta_c$",
        ylabel=r"Difference ($\Delta_c$)",
        legend=False)
    yt, ytl = plt.yticks()

    # combine the two regression means
    ax_final = plt.subplot(2, 2, 2)
    regression(
        R, S, Ri, Si,
        R, model.S_mean, model.S_var,
        title=r"Final GPR for $S$",
        xlabel=None,
        ylabel=None,
        legend=False)
    sg.no_xticklabels()

    # align y-axis labels
    sg.align_ylabels(-0.12, ax_S, ax_logS)
    sg.set_ylabel_coords(-0.16, ax=ax_Dc)
    # sync y-axis limits
    lim = (-0.2, 1.1)
    ax_S.set_ylim(*lim)
    ax_logS.set_ylim(*lim)
    ax_final.set_ylim(*lim)

    # overall figure settings
    sg.set_figsize(9, 5)
    plt.subplots_adjust(
        wspace=0.3, hspace=0.4, left=0.1,
        right=0.95, bottom=0.15, top=0.83)
    plt.suptitle("Bayesian Quadrature Regression", fontsize=16)


def bq_regression_all(model):
    plt.figure()

    R = np.mean([m.R for m in model], axis=0)
    S = [m.S for m in model]
    Sm = np.mean(S, axis=0)
    Ss = scipy.stats.sem(S, axis=0)
    Sl = Sm - Ss
    Su = Sm + Ss
    S = Sm
    plt.fill_between(R, Sl, Su, color='k', alpha=0.2)

    Ri = None
    Si = None

    S_mean = [m.S_mean for m in model]
    Sm = np.mean(S_mean, axis=0)
    Ss = scipy.stats.sem(S_mean, axis=0)
    S_mean = Sm
    S_var = Ss

    regression(
        R, S, Ri, Si,
        R, S_mean, S_var,
        title="Bayesian Quadrature regression for $S$")

    plt.ylim(0, 1.05)


def vm_regression(model):
    plt.figure()
    R = model.R
    S = model.S
    Ri = model.Ri
    Si = model.Si
    S_mean = model.S_mean
    S_var = model.S_var
    regression(
        R, S, Ri, Si,
        R, S_mean, S_var,
        title="Von Mises regression for $S$")
    plt.ylim(0, 1.05)


def vm_regression_all(models):
    plt.figure()

    R = np.mean([m.R for m in models], axis=0)
    S = [m.S for m in models]
    Sm = np.mean(S, axis=0)
    Ss = scipy.stats.sem(S, axis=0)
    Sl = Sm - Ss
    Su = Sm + Ss
    S = Sm
    plt.fill_between(R, Sl, Su, color='k', alpha=0.2)

    Ri = None
    Si = None

    S_mean = [m.S_mean for m in models]
    Sm = np.mean(S_mean, axis=0)
    Ss = scipy.stats.sem(S_mean, axis=0)
    S_mean = Sm
    S_var = Ss

    regression(
        R, S, Ri, Si,
        R, S_mean, S_var,
        title="Von Mises regression for $S$")

    plt.ylim(0, 1.05)


def li_regression(model):
    plt.figure()
    R = model.R
    S = model.S
    Ri = model.Ri
    Si = model.Si
    S_mean = model.S_mean
    S_var = model.S_var
    regression(
        R, S, Ri, Si,
        R, S_mean, S_var,
        title="Linear interpolation for $S$")
    plt.ylim(0, 1.05)


def li_regression_all(models):
    plt.figure()

    R = np.mean([m.R for m in models], axis=0)
    S = [m.S for m in models]
    Sm = np.mean(S, axis=0)
    Ss = scipy.stats.sem(S, axis=0)
    Sl = Sm - Ss
    Su = Sm + Ss
    S = Sm
    plt.fill_between(R, Sl, Su, color='k', alpha=0.2)

    Ri = None
    Si = None

    S_mean = [m.S_mean for m in models]
    Sm = np.mean(S_mean, axis=0)
    Ss = scipy.stats.sem(S_mean, axis=0)
    S_mean = Sm
    S_var = Ss

    regression(
        R, S, Ri, Si,
        R, S_mean, S_var,
        title="Linear interpolation for $S$")

    plt.ylim(0, 1.05)


def likelihood(model):
    plt.figure()
    R = model.R
    S = model.S
    regression(
        R, S, None, None, None, None, None,
        title="Likelihood function",
        legend=False)

    plt.ylim(0, 1.05)


def likelihood_all(models):
    plt.figure()

    R = np.mean([m.R for m in models], axis=0)
    S = [m.S for m in models]
    Sm = np.mean(S, axis=0)
    Ss = scipy.stats.sem(S, axis=0)
    Sl = Sm - Ss
    Su = Sm + Ss
    S = Sm
    plt.fill_between(R, Sl, Su, color='k', alpha=0.2)

    regression(
        R, S, None, None, None, None, None,
        title="Likelihood function",
        legend=False)

    plt.ylim(0, 1.05)


def model_rotations(models):
    fig, axes = plt.subplots(2, len(models), sharex=True, sharey=True)
    if len(models) == 1:
        ax0 = axes[1]
    else:
        ax0 = axes[1][0]

    ax0.set_xticks([0, np.pi / 4., np.pi / 2., 3 * np.pi / 4., np.pi])
    ax0.set_xticklabels([0, 45, 90, 135, 180])
    ax0.set_xlim(-np.pi/16, np.pi+(np.pi/16))

    ax0.set_yticks([0, 25, 50, 75, 100])
    ax0.set_yticklabels([0, 90, 180, 270, 360])
    ax0.set_ylim(-5, 105)

    for i in xrange(len(models)):
        if len(models) == 1:
            tax = axes[0]
            bax = axes[1]
        else:
            tax = axes[0][i]
            bax = axes[1][i]

        model = models[i]
        tax.set_title(model, fontsize=12)
        bax.set_xlabel(r"True rotation ($R$)")
        if i == 0:
            tax.set_ylabel(r"All rotations")
            bax.set_ylabel(r"Mean correct rotations")
        for ax in (tax, bax):
            sg.outward_ticks(ax=ax)
            sg.clear_right(ax=ax)
            sg.clear_top(ax=ax)

    sg.set_figsize(8, 5.5)
    plt.subplots_adjust(
        wspace=0.1, top=0.9, bottom=0.2,
        hspace=0.4, left=0.1, right=0.95)

    x = np.linspace(-np.pi / 16, np.pi + (np.pi / 16.), 100)

    def plot_linreg(i, linreg):
        if len(models) == 1:
            tax = axes[0]
            bax = axes[1]
        else:
            tax = axes[0][i]
            bax = axes[1][i]

        y = x * linreg[0] + linreg[1]
        for ax in (tax, bax):
            ax.plot(x, y, 'k--', dashes=(2, 2), label='linear regression')

    def plot_h0(i, mean, std):
        if len(models) == 1:
            bax = axes[1]
        else:
            bax = axes[1][i]

        y = np.ones(x.shape) * mean
        yl = y - std
        yu = y + std
        bax.fill_between(x, yl, yu, color='c', alpha=0.2)
        bax.plot(x, y, 'b-', label='"different" pairs', linewidth=2)

    def plot_h1(i, x, y, ymean, yerr):
        if len(models) == 1:
            tax = axes[0]
            bax = axes[1]
        else:
            tax = axes[0][i]
            bax = axes[1][i]

        for ia, ang in enumerate(x):
            tax.plot(
                np.ones(y[ia].size) * ang,
                y[ia],
                marker='o',
                markersize=4,
                alpha=0.3,
                linestyle='',
                color='k')

        bax.errorbar(
            x, ymean, yerr=yerr,
            marker='o',
            markersize=7,
            linestyle='',
            color='k',
            label='"same" pairs')

    def plot_legend():
        if len(models[1:]) == 1:
            ax = axes[1]
        else:
            ax = axes[1][1]
        leg = ax.legend(
            loc='upper center',
            frameon=True,
            numpoints=1,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.34),
            ncol=3)
        leg.get_frame().set_edgecolor('white')
        leg.get_frame().set_color('#CCCCCC')
        ax.set_zorder(100)

    funcs = {
        'linreg': plot_linreg,
        'h0': plot_h0,
        'h1': plot_h1,
        'legend': plot_legend
    }
    return fig, axes, funcs


def model_z_accuracy(models):
    fig, axes = plt.subplots(1, len(models), sharex=True, sharey=True)
    if len(models) == 1:
        ax0 = axes
    else:
        ax0 = axes[0]

    xmin = -0.01
    xmax = 0.19
    ymin = -0.02
    ymax = 0.52

    ax0.set_ylabel(r"Estimated $Z$")
    ax0.set_ylim(ymin, ymax)
    ax0.set_xlim(xmin, xmax)

    ticks = [0.0, 0.06, 0.12, 0.18]
    ticklabels = ["%.2f" % x for x in ticks]
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)

    for i in xrange(len(models)):
        if len(models) == 1:
            ax = axes
        else:
            ax = axes[i]
        model = models[i]
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("True $Z$")

        sg.outward_ticks(ax=ax)
        sg.clear_right(ax=ax)
        sg.clear_top(ax=ax)

        ax.plot([xmin, xmax], [xmin, xmax], 'k--', dashes=(2, 2))

    sg.set_figsize(8, 2.7)
    plt.subplots_adjust(
        wspace=0.1, top=0.9, bottom=0.2,
        left=0.1, right=0.95)

    def plot_h0(i, x, y, xerr, yerr, ih0):
        if len(models) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.errorbar(
            x[ih0], y[ih0],
            xerr=xerr[ih0].T if xerr is not None else None,
            yerr=yerr[ih0].T if yerr is not None else None,
            fmt='o',
            label='"different" pairs',
            alpha=0.5,
            color='c')

    def plot_h1(i, x, y, xerr, yerr, ih1):
        if len(models) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.errorbar(
            x[ih1], y[ih1],
            xerr=xerr[ih1].T if xerr is not None else None,
            yerr=yerr[ih1].T if yerr is not None else None,
            fmt='D',
            label=r'"same" pairs',
            alpha=0.5,
            color='m')

    def plot_legend():
        if len(models) == 1:
            ax = axes
        else:
            ax = axes[1]
        leg = ax.legend(
            loc='upper center',
            frameon=True,
            numpoints=1,
            fontsize=12)
        leg.get_frame().set_edgecolor('white')
        leg.get_frame().set_color('#CCCCCC')

    funcs = {
        'h0': plot_h0,
        'h1': plot_h1,
        'legend': plot_legend
    }
    return fig, axes, funcs
