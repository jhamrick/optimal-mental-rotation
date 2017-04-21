#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util
import seaborn as sns


def plot(results_path, fig_path):
    order = ['oc', 'th', 'hc', 'bq', 'bqp']
    titles = {
        'oc': "Oracle",
        'th': "Threshold",
        'hc': "HC",
        'bq': "BQ (equal prior)",
        'bqp': "BQ (unequal prior)"
    }

    results = pd.read_csv(
        results_path.joinpath("response_time_means.csv"))\
        .set_index(['stimulus', 'theta', 'flipped', 'model'])['median']\
        .groupby(level='model')\
        .apply(util.zscore)\
        .unstack('model')

    corrs = pd.read_csv(
        results_path.joinpath("response_time_corrs.csv"))\
        .set_index(['model', 'flipped'])\
        .stack()\
        .unstack(['model', 'flipped'])\
        .xs('all', level='flipped', axis=1)

    fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)

    for i, key in enumerate(order):
        ax = axes.flat[i]
        xmean = results[key]
        ymean = results['exp']
        corr = util.report_pearson.format(**dict(corrs[key]))

        ax.plot([-4, 4], [-4, 4], ls='--', color='#666666', lw=1)
        ax.plot(xmean, ymean, '.', alpha=0.5, color='#333333')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("Model")

        ax.set_title(titles[key])
        ax.text(4, -4, corr, horizontalalignment="right", verticalalignment="bottom", fontsize=8)

    axes[0, 0].set_ylabel("Human")
    axes[1, 0].set_ylabel("Human")

    axes[1, 2].spines['left'].set_color('none')
    axes[1, 2].spines['bottom'].set_color('none')

    sns.despine()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)

    pths = [fig_path.joinpath("response_time_scatters.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
