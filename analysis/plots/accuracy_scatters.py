#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):
    order = ['hc', 'bq', 'bqp']
    titles = {
        'exp': "Human",
        'hc': "HC",
        'bq': "BQ (equal prior)",
        'bqp': "BQ (unequal prior)"
    }

    results = pd.read_csv(
        results_path.joinpath("accuracy_means.csv"))\
        .set_index(['stimulus', 'theta', 'flipped', 'model'])['median']\
        .unstack(['model', 'flipped']) * 100

    fig, axes = plt.subplots(1, len(order), sharey=True, sharex=True)

    for i, key in enumerate(order):
        ax = axes[i]
        for flipped in ['same', 'flipped']:
            ax.plot(
                results[(key, flipped)],
                results[('exp', flipped)],
                '.', alpha=0.8, label=flipped)

        ax.set_xlabel("Model accuracy", fontsize=14)
        ax.set_title(titles[key], fontsize=14)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    axes[0].set_ylabel("Human accuracy", fontsize=14)
    axes[0].set_xlim(-5, 105)
    axes[0].set_ylim(45, 105)
    axes[0].legend(loc=0, numpoints=1)

    fig.set_figheight(3)
    fig.set_figwidth(8)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("accuracy_scatters.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
