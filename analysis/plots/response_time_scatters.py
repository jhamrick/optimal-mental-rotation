#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)
    order = ['oc', 'th', 'hc', 'bq']
    titles = {
        'exp': "Human",
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "Hill climbing",
        'bq': "Bayesian quadrature"
    }

    response_means = {}
    for model in order + ['exp']:
        df = data[model]
        y = df[df['correct']].groupby(
            ['stimulus', 'modtheta', 'flipped'])['ztime']
        response_means[model] = y.apply(util.bootstrap).unstack(-1)['median']
    response_means = pd.DataFrame(response_means).unstack('flipped')

    fig, axes = plt.subplots(1, len(order), sharey=True, sharex=True)

    for i, model in enumerate(order):
        ax = axes[i]

        for key in ('flipped', 'same'):
            ax.plot(
                response_means[model][key],
                response_means['exp'][key],
                '.', alpha=0.8, label=key)

        ax.set_xlabel("Model response time", fontsize=14)
        ax.set_title(titles[model], fontsize=14)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    axes[0].set_ylabel("Human response time", fontsize=14)
    axes[0].set_xlim(-2, 3)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].legend(loc=0, numpoints=1)

    fig.set_figheight(3)
    fig.set_figwidth(14)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_time_scatters.%s" % ext)
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
