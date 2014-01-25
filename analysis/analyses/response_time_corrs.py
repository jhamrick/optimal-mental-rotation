#!/usr/bin/env python

import pandas as pd
import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'bq', 'hc', 'oc', 'th']

    response_means = {}
    for key in keys:
        df = data[key]
        y = df[df['correct']].groupby(
            ['stimulus', 'modtheta', 'flipped'])['ztime']
        response_means[key] = y.apply(util.bootstrap).unstack(-1)['median']
    response_means = pd.DataFrame(response_means)

    pth = results_path.joinpath("response_time_corrs.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for key in keys:
            if key == 'exp':
                continue

            corr = dict(util.bootcorr(
                response_means['exp'],
                response_means[key],
                nsamples=5000,
                method='spearman'))

            print "exp v. %s: %s" % (key, util.report_spearman.format(**corr))
            cmd = util.newcommand(
                "Exp%sTimeCorr" % key.capitalize(),
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
