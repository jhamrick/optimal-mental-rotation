#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "overall_response_time.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for name in sorted(data.keys()):
        # overall mean
        mean = data[name]['time'].mean()
        means = data[name]\
            .groupby(['stimulus', 'theta', 'flipped'])['time']\
            .mean()
        min = means.min()
        max = means.max()
        results[name] = pd.Series({'mean': mean, 'min': min, 'max': max})

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
