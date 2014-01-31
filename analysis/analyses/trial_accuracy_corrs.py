#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "trial_accuracy_corrs.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'expA', 'expB']

    means = pd.read_csv(results_path.joinpath("trial_accuracy_means.csv"))

    results = {}
    for key in keys:
        df = means.groupby('model').get_group(key)
        trials = df['trial']
        accuracy = df['median']
        corr = util.bootcorr(trials, accuracy)
        results[key] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
