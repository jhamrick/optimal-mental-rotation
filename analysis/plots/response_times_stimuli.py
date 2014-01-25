#!/usr/bin/env python

import matplotlib.pyplot as plt
import util
from path import path


def plot(key, data, fig_path):
    fig, axes = plt.subplots(4, 5, sharey=True)

    df = data[key]

    for i, (stim, sdf) in enumerate(df[df['correct']].groupby('stimulus')):
        ax = axes.flat[i]

        for flipped, df2 in sdf.groupby('flipped'):
            time = df2.groupby('modtheta')['ztime']
            stats = time.apply(util.bootstrap).unstack(1)
            lower = stats['median'] - stats['lower']
            upper = stats['upper'] - stats['median']
            ax.errorbar(
                stats.index, stats['median'],
                yerr=[lower, upper],
                label=flipped, lw=3)

        ax.hlines(0, -10, 190, color='k', linestyle='--')
        ax.set_xlabel("Rotation", fontsize=14)
        ax.set_xticks([0, 60, 120, 180])
        ax.set_xlim(-10, 190)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_title("Stim %s" % stim)
        ax.set_ylim(-2, 2)

    fig.set_figheight(8)
    fig.set_figwidth(10)

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("response-times-stimuli-%s.%s" % (key, ext))
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

    for key in ['exp', 'th', 'hc', 'bq']:
        print plot(key, data, fig_path)
