#!/usr/bin/env python

import numpy as np
import util
import pandas as pd
from path import path


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
        print "%s:\t%.2f [%.2f, %.2f]" % (name, mean, min, max)

    results = pd.DataFrame.from_dict(results, orient='index')
    pth = results_path.joinpath("response_time.csv")
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    print run(data, results_path, seed)
