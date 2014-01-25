#!/usr/bin/env python

import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)

    pth = results_path.joinpath("accuracy.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for name, df in data.iteritems():
            a = dict(util.beta(df['correct']) * 100)

            print "%s:\t %s" % (name, util.report_percent.format(**a))
            cmd = util.newcommand(
                "%sAccuracy" % name.capitalize(),
                util.latex_percent.format(**a))

            fh.write(cmd)

    return pth

if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
