#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "theta_accuracy.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for key, df in data.iteritems():
        df = data[key]
        for flipped, fdf in df.groupby('flipped'):
            correct = fdf.groupby('modtheta')['correct']
            stats = correct.apply(util.beta).unstack(1) * 100
            results[(key, flipped)] = stats.stack()

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['model', 'flipped'])
    results.columns = pd.MultiIndex.from_tuples(
        results.columns, names=['modtheta', 'stat'])
    results = results.stack('modtheta')
    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
