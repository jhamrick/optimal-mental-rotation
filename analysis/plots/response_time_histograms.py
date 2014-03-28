#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
import pickle
import numpy as np


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

    fig, axes = plt.subplots(1, len(order), sharey=True)
    for i, key in enumerate(order):
        ax = axes[i]
        bins = 200

        if key == 'exp':
            hist, edges = np.histogram(
                times[key] / 1000., bins=bins,
                range=(0, 20))
        else:
            hist, edges = np.histogram(
                times[key], bins=bins,
                range=(0, bins))

        edges = edges[:101]
        hist = hist[:100]
        hist = hist * 100 / float(len(times[key]))
        width = edges[1] - edges[0]
        ax.bar(edges[:-1], hist, width=width, color='k')

        ax.set_xlim(0, edges[-1])
        ax.set_title(titles[key], fontsize=14)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        if key == 'exp':
            ax.set_xlabel("RT (seconds)", fontsize=14)
        else:
            ax.set_xlabel("Number of actions", fontsize=14)
    axes[0].set_ylabel("Percent", fontsize=14)

    fig.set_figheight(2.5)
    fig.set_figwidth(16)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_time_histograms.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
