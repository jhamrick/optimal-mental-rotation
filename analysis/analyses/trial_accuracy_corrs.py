#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "trial_accuracy_corrs.csv"
texname = "trial_accuracy_corrs.tex"


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'expA', 'expB']

    means = pd.read_csv(results_path.joinpath("trial_accuracy_means.csv"))

    results = {}
    for key in keys:
        df = means.groupby('model').get_group(key)
        trials = df['trial']
        accuracy = df['median']
        corr = util.bootcorr(trials, accuracy, method='spearman')
        results[key] = corr

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for model, stats in results.iterrows():
            cmd = util.newcommand(
                "%sTrialAccuracyCorr" % model.capitalize(),
                util.latex_spearman.format(**dict(stats)))
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
