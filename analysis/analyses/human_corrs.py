#!/usr/bin/env python

import pandas as pd
import numpy as np
import util
import scipy.stats

filename = "human_corrs.csv"
texname = "human_corrs.tex"


def run(data, results_path, seed):
    np.random.seed(seed)

    exp = data['exp'].set_index(['pid', 'trial'])

    pids = pd.Series(exp.index.get_level_values('pid')).drop_duplicates()
    pids.sort()
    pids = pids.reset_index(drop=True)

    n = len(pids)
    m = 100
    time_corrs = np.empty(m)
    acc_corrs = np.empty(m)

    for i in xrange(m):
        idx = np.arange(n)
        np.random.shuffle(idx)

        p0 = pids[idx[:int(n / 2)]]
        p1 = pids[idx[int(n / 2):]]

        df0 = exp.drop(p0, axis=0, level='pid').reset_index()
        df1 = exp.drop(p1, axis=0, level='pid').reset_index()

        tm0 = df0[df0['correct']]\
            .groupby(['stimulus', 'theta', 'flipped'])['time']\
            .apply(lambda x: 1. / (1. / x).mean())
        tm1 = df1[df1['correct']]\
            .groupby(['stimulus', 'theta', 'flipped'])['time']\
            .apply(lambda x: 1. / (1. / x).mean())

        am0 = df0\
            .groupby(['stimulus', 'theta', 'flipped'])['correct']\
            .apply(util.beta).unstack(-1)['median']
        am1 = df1\
            .groupby(['stimulus', 'theta', 'flipped'])['correct']\
            .apply(util.beta).unstack(-1)['median']

        means = pd.DataFrame({'tm0': tm0, 'tm1': tm1, 'am0': am0, 'am1': am1})
        time_corrs[i] = scipy.stats.pearsonr(means['tm0'], means['tm1'])[0]
        acc_corrs[i] = scipy.stats.pearsonr(means['am0'], means['am1'])[0]

    time_stats = np.percentile(time_corrs, [2.5, 50, 97.5])
    acc_stats = np.percentile(acc_corrs, [2.5, 50, 97.5])

    results = pd.DataFrame({
        'time': time_stats,
        'accuracy': acc_stats
    }, index=['lower', 'median', 'upper']).T.reset_index()
    results['model'] = 'exp'
    results = results\
        .rename(columns={'index': 'measure'})\
        .set_index(['model', 'measure'])

    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for (model, measure), stats in results.iterrows():
            cmd = util.newcommand(
                "Exp%sCorr" % measure.capitalize(),
                util.latex_pearson.format(**dict(stats)))
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
