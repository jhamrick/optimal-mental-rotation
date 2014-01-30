#!/usr/bin/env python

import numpy as np
import util
import pandas as pd
from path import path


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for name in sorted(data.keys()):
        df = data[name]
        a = util.beta(df['correct']) * 100
        results[name] = a
        print "%s:\t %s" % (name, util.report_percent.format(**dict(a)))

    results = pd.DataFrame.from_dict(results, orient='index')
    pth = results_path.joinpath("accuracy.csv")
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
