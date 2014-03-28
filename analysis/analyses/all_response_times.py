#!/usr/bin/env python

import util
import pickle

filename = "all_response_times.pkl"


def run(data, results_path, seed):
    results = {}
    for key, df in data.iteritems():
        results[key] = df[df['correct']]['time']

    pth = results_path.joinpath(filename)
    with open(pth, "w") as fh:
        pickle.dump(results, fh)

    return pth

if __name__ == "__main__":
    util.run_analysis(run)
