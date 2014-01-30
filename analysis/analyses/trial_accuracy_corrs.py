#!/usr/bin/env python

import numpy as np
import util
import pandas as pd
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'expA', 'expB']

    results = {}
    for key in keys:
        df = data[key]
        trials = df['trial'].drop_duplicates()
        trials.sort()
        accuracy = df.groupby('trial')['correct']\
                     .apply(util.beta)\
                     .unstack(-1)['median']

        corr = util.bootcorr(trials, accuracy)
        results[key] = corr

        print "%s: %s" % (key, util.report_spearman.format(**dict(corr)))

    results = pd.DataFrame.from_dict(results, orient='index')
    pth = results_path.joinpath("trial_accuracy_corrs.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_human(version, data_path)[1]
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
