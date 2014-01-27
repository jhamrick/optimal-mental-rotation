#!/usr/bin/env python

import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)

    pth = results_path.joinpath("response_time.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for name in sorted(data.keys()):
            # overall mean
            mean = data[name]['time'].mean()
            means = data[name]\
                .groupby(['stimulus', 'theta', 'flipped'])['time']\
                .mean()
            min = means.min()
            max = means.max()

            print "%s:\t%.2f [%.2f, %.2f]" % (name, mean, min, max)
            fh.write(util.newcommand(
                "%sTime" % name.capitalize(), "%.2f" % mean))
            fh.write(util.newcommand(
                "%sTimeMin" % name.capitalize(), "%.2f" % min))
            fh.write(util.newcommand(
                "%sTimeMax" % name.capitalize(), "%.2f" % max))

    return pth

if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
