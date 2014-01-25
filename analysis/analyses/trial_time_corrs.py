#!/usr/bin/env python

import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)

    pth = results_path.joinpath("trial_time_corrs.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for key in sorted(data.keys()):
            df = data[key]
            trials = df[df['correct']]['trial'].drop_duplicates()
            trials.sort()
            times = df[df['correct']]\
                .groupby('trial')['ztime']\
                .apply(util.bootstrap)\
                .unstack(-1)['median']

            corr = dict(util.bootcorr(trials, times))

            print "%s: %s" % (key, util.report_spearman.format(**corr))
            cmd = util.newcommand(
                "%sTrialTimeCorr" % key.capitalize(),
                util.latex_spearman.format(**corr))
            fh.write(cmd)

    return pth

if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_human(version, data_path)[1]
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
