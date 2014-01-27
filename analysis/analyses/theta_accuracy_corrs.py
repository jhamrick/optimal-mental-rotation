#!/usr/bin/env python

import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'hc', 'bq']

    pth = results_path.joinpath("theta_accuracy_corrs.tex")
    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for key in keys:
            for flipped, df in data[key].groupby('flipped'):
                thetas = df['modtheta']
                accuracy = df['correct']
                # x = df.groupby(['stimulus', 'modtheta'])['correct']\
                #       .apply(util.beta)\
                #       .unstack(-1)['median']\
                #       .reset_index()
                # thetas = x['modtheta']
                # accuracy = x['median']

                corr = dict(util.bootcorr(
                    thetas, accuracy, method='spearman'))

                print "%s (%s): %s" % (
                    key, flipped, util.report_spearman.format(**corr))
                cmd = util.newcommand(
                    "%sThetaAccuracyCorr%s" % (
                        key.capitalize(), flipped.capitalize()),
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
