#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "theta_time.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for key, df in data.iteritems():
        y = df[df['correct']].groupby(
            ['modtheta', 'flipped'])['time']
        results[key] = y.apply(
            lambda x: 1. / util.bootstrap_mean(1. / x))

    results = pd.DataFrame.from_dict(results)
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['modtheta', 'flipped', 'stat'])
    results.columns.name = 'model'
    results = results.stack().unstack('stat')
    results.columns.name = None
    results = results.reset_index()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
