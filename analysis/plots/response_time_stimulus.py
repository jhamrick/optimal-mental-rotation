#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd
from path import path


def plot(results_path, fig_path):
    order = ['exp', 'th', 'bqp']
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

    means = pd.read_csv(
        results_path.joinpath("response_time_means.csv"))\
        .set_index(['stimulus', 'flipped', 'model'])\
        .groupby(level='stimulus').get_group(2)

    fig, axes = plt.subplots(1, len(order), sharex=True)
    for i, key in enumerate(order):
        ax = axes[i]
        df = means.groupby(level='model').get_group(key)

        for flipped, stats in df.groupby(level='flipped'):
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
        ax.set_xlabel("Rotation")
        ax.set_title(titles[key], fontsize=14)

        if key == 'exp':
            ax.set_ylabel("Response time", fontsize=14)
        else:
            ax.set_ylabel("Number of actions", fontsize=14)

    # util.sync_ylims(axes[order.index('bq')], axes[order.index('bqp')])

    for ax in axes.flat:
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.95')

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = axes[0].legend(
        [p0, p1], ["same", "flipped"],
        numpoints=1, fontsize=12,
        loc='lower right')
    frame = leg.get_frame()
    frame.set_edgecolor('#FFFFFF')

    util.sync_ylabel_coords(axes.flat, -0.175)

    fig.set_figheight(3)
    fig.set_figwidth(9)

    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    pths = [fig_path.joinpath("response_time_stimulus.%s" % ext)
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
    print plot(data, fig_path)
