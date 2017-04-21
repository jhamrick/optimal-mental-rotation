#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd
import seaborn as sns


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
        'same': [0.16339869, 0.4449827, 0.69750096],
        'flipped': [0.72848904, 0.1550173, 0.19738562]
    }

    acc_results = pd.read_csv(
        results_path.joinpath("theta_accuracy.csv"))

    fig, axes = plt.subplots(2, 3, sharex=False)
    for i, key in enumerate(order):
        ax = axes.flat[i]
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
        ax.set_ylim(-5, 105)
        ax.set_xticks(np.arange(0, 200, 60))
        ax.set_xlabel("Rotation")
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels([0.0, 0.5, 1.0])
        ax.set_title(titles[key])
        ax.set_ylabel("Accuracy")

    p0 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['same'],
        ec=colors['same'])
    p1 = plt.Rectangle(
        (0, 0), 1, 1,
        fc=colors['flipped'],
        ec=colors['flipped'])

    leg = axes[0, 0].legend(
        [p0, p1], ["same", "flipped"],
        numpoints=1,
        loc='lower right',
        bbox_to_anchor=[1.1, 0])

    sns.despine()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)

    pths = [fig_path.joinpath("accuracy.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
