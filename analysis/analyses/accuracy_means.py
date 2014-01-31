#!/usr/bin/env python

import numpy as np
import util
import pandas as pd


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for key, df in data.iteritems():
        y = df.groupby(['stimulus', 'modtheta', 'flipped'])['correct']
        results[key] = y.apply(util.beta)

    results = pd.DataFrame.from_dict(results)
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['stimulus', 'modtheta', 'flipped', 'stat'])
    results.columns.name = 'model'
    results = results.stack().unstack('stat')

    pth = results_path.joinpath("accuracy_means.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
