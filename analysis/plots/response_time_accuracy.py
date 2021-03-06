#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd


def plot(results_path, fig_path):
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
        'same': [ 0.16339869,  0.4449827 ,  0.69750096],
        'flipped': [ 0.72848904,  0.1550173 ,  0.19738562]
    }

    time_results = pd.read_csv(
        results_path.joinpath("theta_time.csv"))
    acc_results = pd.read_csv(
        results_path.joinpath("theta_accuracy.csv"))

    fig, axes = plt.subplots(2, len(order), sharex=True)
    for i, key in enumerate(order):
        ax = axes[0, i]
        df = time_results.groupby('model').get_group(key)

        for flipped, stats in df.groupby('flipped'):
            if key == 'exp':
                median = stats['median'] / 1000.
                lower = (stats['median'] - stats['lower']) / 1000.
                upper = (stats['upper'] - stats['median']) / 1000.
            else:
                median = stats['median']
                lower = stats['median'] - stats['lower']
                upper = stats['upper'] - stats['median']

            ax.errorbar(
                stats['modtheta'], median,
                yerr=[lower, upper], lw=3,
                color=colors[flipped],
                ecolor=colors[flipped])

        ax.set_xticks(np.arange(0, 200, 30))
        ax.set_xlim(-10, 190)

        ax.set_title(titles[key], fontsize=14)

        if key == 'exp':
            ax.set_ylabel("Response time", fontsize=14)
        else:
            ax.set_ylabel("# actions", fontsize=14)

    util.sync_ylims(axes[0, order.index('bq')], axes[0, order.index('bqp')])

    for i, key in enumerate(order):
        ax = axes[1, i]
        df = acc_results.groupby('model').get_group(key)

        for flipped, stats in df.groupby('flipped'):
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats['modtheta'], stats['median'],
                yerr=[lower, upper], lw=3,
                color=colors[flipped],
                ecolor=colors[flipped])

        ax.set_xlim(-10, 190)
        ax.set_ylim(25, 105)
        ax.set_xticks(np.arange(0, 200, 30))
        ax.set_xlabel("Rotation", fontsize=14)

        ax.set_ylabel("Accuracy", fontsize=14)

    for ax in axes.flat:
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = axes[1, 1].legend(
        [p0, p1], ["\"same\" pairs", "\"flipped\" pairs"],
        numpoints=1, fontsize=12,
        loc='lower center',
        title='Stimuli')
    frame = leg.get_frame()
    frame.set_edgecolor('#FFFFFF')

    util.sync_ylabel_coords(axes.flat, -0.175)

    fig.set_figheight(4)
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
    util.make_plot(plot)
