#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd
import seaborn as sns


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
        'same': [ 0.16339869,  0.4449827 ,  0.69750096],
        'flipped': [ 0.72848904,  0.1550173 ,  0.19738562]
    }

    means = pd.read_csv(
        results_path.joinpath("theta_time_stimulus.csv"))\
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

        ax.set_xticks(np.arange(0, 200, 60))
        ax.set_xlim(-10, 190)
        ax.set_xlabel("Rotation")
        ax.set_title(titles[key])

        if key == "exp":
            ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3])
            ax.set_ylabel("Rotation")
        elif key == "th":
            ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
            ax.set_ylabel("# Steps")
        elif key == "bqp":
            ax.set_yticks([10, 15, 20, 25])
            ax.set_ylabel("# Steps")

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = axes.flat[0].legend(
        [p0, p1], ["same", "flipped"],
        numpoints=1,
        loc='lower right',
        bbox_to_anchor=[1.1, 0])

    fig.set_size_inches(6, 2)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)

    pths = [fig_path.joinpath("response_time_stimulus.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
