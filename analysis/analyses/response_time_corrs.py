#!/usr/bin/env python

import pandas as pd
import numpy as np
import util
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'oc', 'th', 'hc', 'bq', 'bqp']

    response_means = {}
    for key in keys:
        df = data[key]
        y = df[df['correct']].groupby(
            ['stimulus', 'modtheta', 'flipped'])['time']
        response_means[key] = y.mean()
    response_means = pd.DataFrame(response_means)

    results = {}
    for key in keys:
        if key == 'exp':
            continue

        corr = util.bootcorr(
            response_means['exp'],
            response_means[key],
            method='pearson')

        print "exp v. %s: %s" % (key, util.report_pearson.format(**dict(corr)))
        results[key] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    pth = results_path.joinpath("response_time_corrs.csv")
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
