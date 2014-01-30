#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)

    def makecorr(df):
        x = df.groupby(['stimulus', 'modtheta'])['correct']\
              .mean()\
              .reset_index()
        thetas = x['modtheta']
        acc = 1 - x['correct']
        corr = util.bootcorr(thetas, acc, method="spearman")
        return corr

    keys = ['exp', 'oc', 'th', 'hc', 'bq', 'bqp']
    corrs = {}
    for key in keys:
        for flipped, df in data[key].groupby('flipped'):
            corrs[(key, flipped)] = makecorr(df)

    df = pd.DataFrame.from_dict(corrs, orient='index').fillna(0)
    df.index = pd.MultiIndex.from_tuples(df.index, names=['model', 'stimuli'])
    df = df.unstack('stimuli').reindex(keys).stack('stimuli')

    fig, ax = plt.subplots()
    width = 1
    offset = width * 5 / 2.
    colors = {
        'same': '#ff5555',
        'flipped': '#5555ff'
    }
    titles = {
        'exp': "Human",
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ\n(equal)",
        'bqp': "BQ\n(unequal)"
    }

    order = ['same', 'flipped']
    for flipped, y in df.groupby(level='stimuli'):
        i = order.index(flipped)
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

    ax.set_title("Correlation between rotation and accuracy", fontsize=14)
    ax.set_ylabel(r"Spearman correlation ($r_s$)", fontsize=14)

    pths = [fig_path.joinpath("theta_accuracy_corrs.%s" % ext)
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
