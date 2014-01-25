#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)
    fig, axes = plt.subplots(1, 5, sharey=True)

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
        if 'nstep' in df:
            bins = df['nstep'].ptp() + 1
        else:
            bins = 100

        ax.hist(df['time'], bins=bins, normed=True, color='k')
        ax.set_xlabel("Response time", fontsize=14)
        ax.set_title(titles[key], fontsize=14)

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks()

    axes[0].set_ylabel("Fraction of responses", fontsize=14)

    fig.set_figheight(3)
    fig.set_figwidth(16)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_time_histograms.%s" % ext)
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
