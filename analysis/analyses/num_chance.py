#!/usr/bin/env python

import numpy as np
import util
import pandas as pd

filename = "num_chance.csv"
texname = "num_chance.tex"

words = [
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'ten'
]


def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    exclude = ['expA', 'expB', 'gs']
    for key, df in data.iteritems():
        if key in exclude:
            continue
        y = df.groupby(['stimulus', 'theta', 'flipped'])['correct']
        alpha = 0.05 / len(y.groups)
        chance = y.apply(util.beta, [alpha]).unstack(-1)[alpha] <= 0.5
        results[key] = chance

    results = pd.DataFrame.from_dict(results).stack().reset_index()
    results.columns = ['stimulus', 'theta', 'flipped', 'model', 'chance']

    pth = results_path.joinpath(filename)
    results.set_index("stimulus").to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for model, chance in results.groupby('model')['chance']:
            num = chance.sum()
            if num < len(words):
                num = words[num]
            cmd = util.newcommand(
                "%sNumChance" % model.capitalize(), num)
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
