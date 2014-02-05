#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
import pickle
import numpy as np
from path import path


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

    pth = results_path.joinpath("all_response_times.pkl")
    with open(pth, "r") as fh:
        times = pickle.load(fh)

    fig, axes = plt.subplots(1, len(order))
    for i, key in enumerate(order):
        ax = axes[i]

        if key == 'exp':
            bins = 100
        else:
            bins = times[key].ptp() + 1

        ax.hist(np.asarray(times[key]), bins=bins, color='k')
        ax.set_title(titles[key], fontsize=14)

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        if key == 'exp':
            ax.set_xlabel("Response time", fontsize=14)
        else:
            ax.set_xlabel("Number of actions", fontsize=14)

    axes[0].set_ylabel("Number of responses", fontsize=14)

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
    results_path = path(config.get("paths", "results")).joinpath(version)
    fig_path = path(config.get("paths", "figures")).joinpath(version)
    print plot(results_path, fig_path)
