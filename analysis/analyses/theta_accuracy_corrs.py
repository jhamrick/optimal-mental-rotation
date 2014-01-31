#!/usr/bin/env python

import numpy as np
import util
import pandas as pd


def run(data, results_path, seed):
    np.random.seed(seed)

    means = pd.read_csv(results_path.joinpath("accuracy_means.csv"))
    results = {}
    exclude = ['expA', 'expB', 'gs']
    for (model, flipped), df in means.groupby(['model', 'flipped']):
        if model in exclude:
            continue
        results[(model, flipped)] = util.bootcorr(
            df['modtheta'], df['median'], method='spearman')

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['model', 'flipped'])
    pth = results_path.joinpath("theta_accuracy_corrs.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
