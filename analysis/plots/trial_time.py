#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import util
from path import path


def plot(data, fig_path, seed):
    np.random.seed(seed)

    df = data['exp']
    trials = df[df['correct']]['trial'].drop_duplicates()
    trials.sort()
    times = df[df['correct']]\
        .groupby('trial')['ztime']\
        .apply(util.bootstrap)\
        .unstack(-1)['median']

    fig, ax = plt.subplots()
    ax.plot(np.asarray(trials), np.asarray(times), 'k.')
    ax.set_xlim(1, 200)
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Response time (z-scored)", fontsize=14)
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
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_human(version, data_path)[1]
    fig_path = path(config.get("paths", "figures")).joinpath(version)
    seed = config.getint("global", "seed")
    print plot(data, fig_path, seed)
