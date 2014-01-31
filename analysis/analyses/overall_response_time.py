#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "overall_response_time.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for name in sorted(data.keys()):
        results[name] = 1. / util.bootstrap_mean(1. / data[name]['time'])

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
