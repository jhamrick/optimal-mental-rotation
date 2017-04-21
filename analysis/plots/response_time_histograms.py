#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
import pickle
import numpy as np
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

    pth = results_path.joinpath("all_response_times.pkl")
    with open(pth, "r") as fh:
        times = pickle.load(fh)

    fig, axes = plt.subplots(2, 3, sharey=True)
    for i, key in enumerate(order):
        ax = axes.flat[i]
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
        ax.fill_between(edges[:-1], hist, np.zeros_like(hist), color='#666666')

        ax.set_xlim(0, edges[-1])
        ax.set_title(titles[key])

        if key == 'exp':
            ax.set_xlabel("RT (seconds)")
        else:
            ax.set_xlabel("Number of actions")

    axes[0, 0].set_ylabel("Percent")
    axes[1, 0].set_ylabel("Percent")

    sns.despine()

    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)

    pths = [fig_path.joinpath("response_time_histograms.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
