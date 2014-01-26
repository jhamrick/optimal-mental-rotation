#!/usr/bin/env python

import pandas as pd
import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'bq', 'hc']

    accuracy_means = {}
    for key in keys:
        df = data[key]
        y = df.groupby(['stimulus', 'modtheta', 'flipped'])['correct']
        accuracy_means[key] = y.apply(util.beta).unstack(-1)['median']
    accuracy_means = pd.DataFrame(accuracy_means)

    pth = results_path.joinpath("accuracy_corrs.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for key in keys:
            if key == 'exp':
                continue

            corr = dict(util.bootcorr(
                accuracy_means['exp'],
                accuracy_means[key],
                method='spearman'))

            print "exp v. %s: %s" % (key, util.report_spearman.format(**corr))
            cmd = util.newcommand(
                "Exp%sAccuracyCorr" % key.capitalize(),
                util.latex_spearman.format(**corr))
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
