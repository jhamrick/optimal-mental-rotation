#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)

    order = ['exp', 'oc', 'th', 'hc', 'bq']
    titles = {
        'exp': "Human",
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "Hill climbing",
        'bq': "Bayesian quadrature"
    }

    for i, key in enumerate(order):
        ax = axes[i]
        df = data[key]

        for flipped, df2 in df[df['correct']].groupby('flipped'):
            time = df2.groupby('modtheta')['ztime']
            stats = time.apply(util.bootstrap).unstack(1)
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats.index, stats['median'],
                yerr=[lower, upper],
                label=flipped, lw=3)

        ax.hlines(0, -10, 190, color='k', linestyle='--')
        ax.set_xlabel("Rotation", fontsize=14)
        ax.set_xticks(np.arange(0, 200, 20))
        ax.set_xlim(-10, 190)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        ax.set_title(titles[key], fontsize=14)

    axes[0].legend(title="Stimuli", loc=0, frameon=False)
    axes[0].set_ylim(-2, 2)
    fig.set_figheight(3)
    fig.set_figwidth(18)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_times.%s" % ext)
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
