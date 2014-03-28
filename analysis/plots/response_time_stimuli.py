#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
import pandas as pd


def plot_key(key, results_path, fig_path):
    fig, axes = plt.subplots(4, 5, sharey=True, sharex=True)

    means = pd.read_csv(
        results_path.joinpath("theta_time_stimulus.csv"))\
        .set_index(['stimulus', 'flipped', 'model'])\
        .groupby(level='model').get_group(key)

    for i, (stim, sdf) in enumerate(means.groupby(level='stimulus')):
        ax = axes.flat[i]

        for flipped, stats in sdf.groupby(level='flipped'):
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats['modtheta'], stats['median'],
                yerr=[lower, upper],
                label=flipped, lw=3)

        ax.set_xlabel("Rotation")
        ax.set_xticks([0, 60, 120, 180])
        ax.set_xlim(-10, 190)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_title("Stim %s" % stim)

    fig.set_figheight(8)
    fig.set_figwidth(10)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response_time_stimuli_%s.%s" % (key, ext))
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


def plot(results_path, fig_path):
    pths = []
    for key in ['exp', 'th', 'hc', 'bq', 'bqp']:
        pths.extend(plot_key(key, results_path, fig_path))
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
