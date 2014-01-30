#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)

    order = ['exp', 'oc', 'th', 'hc', 'bq', 'bqp']
    titles = {
        'exp': "Human",
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ (equal prior)",
        'bqp': "BQ (unequal prior)"
    }

    colors = {
        'same': 'r',
        'flipped': 'b'
    }

    fig, axes = plt.subplots(2, len(order), sharex=True)
    for i, key in enumerate(order):
        ax = axes[0, i]
        df = data[key]

        for flipped, df2 in df[df['correct']].groupby('flipped'):
            time = df2.groupby('modtheta')['time']
            stats = time.apply(util.bootstrap).unstack(1)
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats.index, stats['median'],
                yerr=[lower, upper], lw=3,
                color=colors[flipped],
                ecolor=colors[flipped])

        ax.set_xticks(np.arange(0, 200, 30))
        ax.set_xlim(-10, 190)

        ax.set_title(titles[key], fontsize=14)

        if key == 'exp':
            ax.set_ylabel("Response time", fontsize=14)
        else:
            ax.set_ylabel("Number of actions", fontsize=14)

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    util.sync_ylims(axes[0, order.index('bq')], axes[0, order.index('bqp')])

    for i, key in enumerate(order):
        ax = axes[1, i]
        df = data[key]

        for flipped, df2 in df.groupby('flipped'):
            correct = df2.groupby('modtheta')['correct']
            stats = correct.apply(util.beta).unstack(1) * 100
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats.index, stats['median'],
                yerr=[lower, upper], lw=3,
                color=colors[flipped],
                ecolor=colors[flipped])

        ax.set_xlim(-10, 190)
        ax.set_ylim(25, 105)
        ax.set_xticks(np.arange(0, 200, 30))
        ax.set_xlabel("Rotation", fontsize=14)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        ax.set_ylabel("Percent correct", fontsize=14)

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = axes[1, 0].legend(
        [p0, p1], ["\"same\" pairs", "\"flipped\" pairs"],
        numpoints=1, fontsize=12,
        loc='lower center',
        title='Stimuli')
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('#FFFFFF')

    util.sync_ylabel_coords(axes.flat, -0.175)

    fig.set_figheight(5)
    fig.set_figwidth(18)

    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    pths = [fig_path.joinpath("response_time_accuracy.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    fig_path = path(config.get("paths", "figures")).joinpath(version)
    seed = config.getint("global", "seed")
    print plot(data, fig_path, seed)