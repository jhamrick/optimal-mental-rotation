#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


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

    fig, axes = plt.subplots(1, len(order), sharey=True, sharex=True)

    for i, key in enumerate(order):
        ax = axes[i]
        xmean = results[key]
        ymean = results['exp']
        corr = util.report_pearson.format(**dict(corrs[key]))

        ax.plot([-4, 4], [-4, 4], ls='--', color='#666666')
        ax.plot(xmean, ymean, '.', alpha=0.9, color='k')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(titles[key], fontsize=14)
        ax.set_xlabel(corr, fontsize=14)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        ax.set_axis_bgcolor('0.95')

    util.sync_xlims(axes[order.index('bq')], axes[order.index('bqp')])

    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_ylabel("Human", fontsize=14)

    fig.set_figheight(3)
    fig.set_figwidth(16)
    plt.draw()
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1)

    pths = [fig_path.joinpath("response_time_scatters.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
