#!/usr/bin/env python

import util
from path import path


def run(data, results_path):
    pth = results_path.joinpath("response_times.tex")

    with open(pth, "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")

        for name, df in data.iteritems():
            t = util.bootstrap(df['time'])

            print "%s:\t%.2f [%.2f, %.2f]" % (
                name, t['median'], t['lower'], t['upper'])

            for stat, val in t.iteritems():
                cmd = util.newcommand(
                    "%sTime%s" % (name.capitalize(), stat.capitalize()),
                    "%.2f" % val)
                fh.write(cmd)

    return pth

if __name__ == "__main__":
    config = util.load_config("config.ini")
    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))
    data = util.load_all(version, data_path)
    results_path = path(config.get("paths", "results")).joinpath(version)
    print run(data, results_path)
