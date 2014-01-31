#!/usr/bin/env python

import pandas as pd
import numpy as np
import util

filename = "response_time_corrs.csv"
texname = "response_time_corrs.tex"


def run(data, results_path, seed):
    np.random.seed(seed)

    means = pd.read_csv(results_path.joinpath("response_time_means.csv"))
    means = means\
        .set_index(['stimulus', 'modtheta', 'flipped', 'model'])['median']\
        .unstack('model')

    results = {}
    exclude = ['exp', 'expA', 'expB', 'gs']
    for key in means:
        if key in exclude:
            continue
        for flipped in ['same', 'flipped']:
            corr = util.bootcorr(
                means['exp'].unstack('flipped')[flipped],
                means[key].unstack('flipped')[flipped],
                method='pearson')
            results[(key, flipped)] = corr

        corr = util.bootcorr(
            means['exp'],
            means[key],
            method='pearson')
        results[(key, 'all')] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index, names=['model', 'flipped'])
    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for (model, flipped), stats in results.iterrows():
            if flipped != 'all':
                continue
            cmd = util.newcommand(
                "%sTimeCorr" % model.capitalize(),
                util.latex_pearson.format(**dict(stats)))
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
