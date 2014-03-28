#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "theta_time_corrs.csv"
texname = "theta_time_corrs.tex"


def run(data, results_path, seed):
    np.random.seed(seed)

    means = pd.read_csv(results_path.joinpath("response_time_means.csv"))
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
    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for (model, flipped), stats in results.iterrows():
            cmd = util.newcommand(
                "%sThetaTimeCorr%s" % (
                    model.capitalize(),
                    flipped.capitalize()),
                util.latex_spearman.format(**dict(stats)))
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
