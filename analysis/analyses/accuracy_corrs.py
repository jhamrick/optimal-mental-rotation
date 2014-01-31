#!/usr/bin/env python

import pandas as pd
import numpy as np
import util


def run(data, results_path, seed):
    np.random.seed(seed)

    means = pd.read_csv(
        results_path.joinpath("accuracy_means.csv"))\
        .set_index(['stimulus', 'modtheta', 'flipped', 'model'])['median']\
        .unstack('model')

    results = {}
    exclude = ['exp', 'expA', 'expB', 'gs']
    for key in means:
        if key in exclude:
            continue
        corr = util.bootcorr(
            means['exp'],
            means[key],
            method='pearson')
        results[key] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath("accuracy_corrs.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
