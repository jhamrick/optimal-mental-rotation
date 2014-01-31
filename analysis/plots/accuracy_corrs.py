#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
from path import path


def plot(results_path, fig_path):
    keys = ['oc', 'th', 'hc', 'bq', 'bqp']
    corrs = pd.read_csv(results_path.joinpath("accuracy_corrs.csv"))\
              .set_index(['model', 'flipped'])\
              .unstack('flipped')\
              .reindex(keys)

    fig, ax = plt.subplots()
    width = 1
    offset = width * 5 / 2.
    colors = {
        'same': '#ff5555',
        'flipped': '#5555ff'
    }
    titles = {
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ\n(equal)",
        'bqp': "BQ\n(unequal)"
    }

    order = ['same', 'flipped']
    for i, flipped in enumerate(order):
        y = corrs.xs(flipped, axis=1, level='flipped')
        median = y['median']
        lerr = median - y['lower']
        uerr = y['upper'] - median
        ax.bar(
            i * width + np.arange(len(median)) * offset,
            median,
            yerr=[lerr, uerr],
            color=colors[flipped],
            ecolor='k',
            width=width,
            edgecolor='none')

    ax.set_ylim(0, 1)
    ax.set_xlim(-width / 2., len(median) * offset)
    ax.set_xticks(np.arange(len(median)) * offset + width)
    ax.set_xticklabels([titles[key] for key in keys], fontsize=10)

    util.clear_right(ax)
    util.clear_top(ax)
    util.outward_ticks(ax)

    ax.set_title("Correlation with human accuracy", fontsize=14)
    ax.set_ylabel(r"Pearson correlation ($r$)", fontsize=14)

    pths = [fig_path.joinpath("accuracy_corrs.%s" % ext)
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
