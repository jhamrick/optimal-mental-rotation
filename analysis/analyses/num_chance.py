#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "num_chance.csv"
texname = "num_chance.tex"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    exclude = ['expA', 'expB', 'gs']
    for key, df in data.iteritems():
        if key in exclude:
            continue
        y = df.groupby(['stimulus', 'theta', 'flipped'])['correct']
        num = (y.apply(util.beta, [0.05]) <= 0.5).sum()
        total = len(y.groups)
        results[key] = pd.Series({'num': num, 'total': total})

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = 'model'
    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for model, stats in results.iterrows():
            cmd = util.newcommand(
                "%sNumChance" % model.capitalize(),
                stats['num'])
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
