#!/usr/bin/env python

import numpy as np
import util
import pandas as pd
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)
    keys = ['exp', 'bq', 'bqp', 'hc']

    results = {}
    for key in keys:
        df = data[key]
        y = df.groupby(['stimulus', 'theta', 'flipped'])['correct']
        num = (y.apply(util.beta, [0.05]) <= 0.5).sum()
        total = len(y.groups)
        results[key] = pd.Series({'num': num, 'total': total})
        print "%s: %d / %d" % (key, num, total)

    results = pd.DataFrame.from_dict(results, orient='index')
    pth = results_path.joinpath("num_chance.csv")
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
