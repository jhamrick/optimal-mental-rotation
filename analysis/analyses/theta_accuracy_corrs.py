#!/usr/bin/env python

import numpy as np
import util
import pandas as pd
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'hc', 'bq', 'bqp']

    results = {}
    for key in keys:
        for flipped, df in data[key].groupby('flipped'):
            x = df.groupby(['stimulus', 'modtheta'])['correct']\
                  .apply(util.beta)\
                  .unstack(-1)['median']\
                  .reset_index()
            thetas = x['modtheta']
            accuracy = x['median']

            corr = util.bootcorr(
                thetas, accuracy, method='spearman')

            print "%s (%s): %s" % (
                key, flipped, util.report_spearman.format(**dict(corr)))
            results[(key, flipped)] = corr

        df = data[key]
        x = df.groupby(['stimulus', 'modtheta'])['correct']\
              .apply(util.beta)\
              .unstack(-1)['median']\
              .reset_index()
        thetas = x['modtheta']
        accuracy = x['median']

        corr = util.bootcorr(
            thetas, accuracy, method='spearman')

        print "%s (all): %s" % (
            key, util.report_spearman.format(**dict(corr)))
        results[(key, 'all')] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['model', 'flipped'])
    pth = results_path.joinpath("theta_accuracy_corrs.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
