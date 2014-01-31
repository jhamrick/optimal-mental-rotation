#!/usr/bin/env python

import numpy as np
import util
import pandas as pd


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'expA', 'expB']

    results = {}
    for key in keys:
        df = data[key]
        trials = df[df['correct']]['trial'].drop_duplicates()
        trials.sort()
        times = df[df['correct']].groupby('trial')['time'].mean()
        corr = util.bootcorr(trials, times, method='spearman')
        results[key] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath("trial_time_corrs.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
