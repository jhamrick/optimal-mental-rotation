#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


def plot(results_path, fig_path):
    keys = ['exp', 'oc', 'th', 'hc', 'bq', 'bqp']
    corrs = pd.read_csv(results_path.joinpath("theta_time_corrs.csv"))\
              .set_index(['model', 'flipped'])\
              .unstack('flipped')\
              .reindex(keys)

    fig, ax = plt.subplots()
    width = 1
    offset = width * 5 / 2.
    colors = {
        'same': '#ff5555',
        'flipped': '#5555ff'
    }
    titles = {
        'exp': "Human",
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ\n(equal)",
        'bqp': "BQ\n(unequal)"
    }

    order = ['same', 'flipped']
    for i, flipped in enumerate(order):
        y = corrs.xs(flipped, axis=1, level='flipped')
        median = y['median']
        lerr = median - y['lower']
        uerr = y['upper'] - median
        ax.bar(
            i * width + np.arange(len(median)) * offset,
            median,
            yerr=[lerr, uerr],
            color=colors[flipped],
            ecolor='k',
            width=width,
            edgecolor='none',
            capsize=0)

    ax.set_ylim(0, 1)
    ax.set_xlim(-width / 2., len(median) * offset)
    ax.set_xticks(np.arange(len(median)) * offset + width)
    ax.set_xticklabels([titles[key] for key in keys], fontsize=10)

    util.clear_right(ax)
    util.clear_top(ax)
    util.outward_ticks(ax)

    ax.set_ylabel(r"Spearman correlation ($r_s$)", fontsize=14)

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = ax.legend(
        [p0, p1], ["\"same\" pairs", "\"flipped\" pairs"],
        numpoints=1, fontsize=12,
        loc='upper right',
        bbox_to_anchor=(1, 1.05))
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('#FFFFFF')

    fig.set_figwidth(5)
    fig.set_figheight(3)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("theta_time_corrs.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
