#!/usr/bin/env python

import util
import pickle


def run(data, results_path, seed):
    results = {}
    for key, df in data.iteritems():
        results[key] = df['time']

    pth = results_path.joinpath("all_response_times.pkl")
    with open(pth, "w") as fh:
        pickle.dump(results, fh)

    return pth

if __name__ == "__main__":
    util.run_analysis(run)
