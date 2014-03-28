#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
import pandas as pd


def plot(results_path, fig_path):
    data = pd.read_csv(
        results_path.joinpath("trial_time_means.csv"))\
        .groupby('model').get_group('exp')\
        .sort('trial')

    trials = data['trial']
    times = data['median']

    fig, ax = plt.subplots()
    ax.plot(trials, times, 'k.')
    ax.set_xlim(1, 200)
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Response time", fontsize=14)
    util.clear_right(ax)
    util.clear_top(ax)
    util.outward_ticks(ax)
    fig.set_figheight(3.5)
    fig.set_figwidth(4.5)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("trial_time.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
