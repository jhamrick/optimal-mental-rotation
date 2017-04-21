#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "trial_time_means.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'expA', 'expB']

    results = {}
    for key in keys:
        df = data[key]
        times = df[df['correct']].groupby('trial')['time']
        results[key] = times.apply(util.bootstrap_logmean)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    results.columns = pd.MultiIndex.from_tuples(
        results.columns, names=['trial', 'stat'])
    results = results.stack('trial')
    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
