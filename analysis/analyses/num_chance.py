#!/usr/bin/env python

import numpy as np
import util
import pandas as pd


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for key, df in data.iteritems():
        y = df.groupby(['stimulus', 'theta', 'flipped'])['correct']
        num = (y.apply(util.beta, [0.05]) <= 0.5).sum()
        total = len(y.groups)
        results[key] = pd.Series({'num': num, 'total': total})

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath("num_chance.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
