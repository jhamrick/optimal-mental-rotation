#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
from path import path


def plot(results_path, fig_path):
    keys = ['oc', 'th', 'hc', 'bq', 'bqp']
    corrs = pd.read_csv(results_path.joinpath("response_time_corrs.csv"))\
              .set_index(['model', 'flipped'])\
              .unstack('flipped')\
              .reindex(keys)

    fig, ax = plt.subplots()
    width = 0.8

    titles = {
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ\n(equal)",
        'bqp': "BQ\n(unequal)"
    }

    y = corrs.xs('all', axis=1, level='flipped')
    median = y['median']
    lerr = median - y['lower']
    uerr = y['upper'] - median
    ax.bar(
        np.arange(len(median)),
        median,
        yerr=[lerr, uerr],
        color='0.5',
        ecolor='k',
        width=width,
        edgecolor='none',
        capsize=0)

    ax.set_ylim(0, 1)
    ax.set_xlim(-width / 2., len(median))
    ax.set_xticks(np.arange(len(median)) + (width / 2.))
    ax.set_xticklabels([titles[key] for key in keys], fontsize=10)

    util.clear_right(ax)
    util.clear_top(ax)
    util.outward_ticks(ax)

    ax.set_title("Human vs. model response time", fontsize=14)
    ax.set_ylabel(r"Pearson correlation ($r$)", fontsize=14)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_time_corrs.%s" % ext)
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
