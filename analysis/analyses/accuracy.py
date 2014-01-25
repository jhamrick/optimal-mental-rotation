#!/usr/bin/env python

import util
from path import path


def run(data, results_path):
    pth = results_path.joinpath("accuracy.tex")

    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for name, df in data.iteritems():
            a = util.beta(df['correct']) * 100

            print "%s:\t%.1f%% [%.1f, %.1f]" % (
                name, a['median'], a['lower'], a['upper'])

            for stat, val in a.iteritems():
                cmd = util.newcommand(
                    "%sAccuracy%s" % (name.capitalize(), stat.capitalize()),
                    "%.1f" % val)
                fh.write(cmd)

    return pth

if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    print run(data, results_path)
