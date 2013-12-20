#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
from mental_rotation import EXP_PATH, STIM_PATH
from mental_rotation.stimulus import Stimulus2D
import logging
import pandas as pd

logger = logging.getLogger("mental_rotation.experiment")


def get_stiminfo(stim_path):
    stim = Stimulus2D.load(stim_path)

    assert len(stim.operations) <= 2
    if len(stim.operations) == 2:
        assert stim.operations[0][0] == "flip"
        assert stim.operations[1][0] == "rotate"
        flipped = True
        theta = stim.operations[1][1]
    elif len(stim.operations) == 1:
        if stim.operations[0][0] == "rotate":
            flipped = False
            theta = stim.operations[0][1]
        elif stim.operations[0][0] == "flip":
            flipped = True
            theta = 0
    elif len(stim.operations) == 0:
        flipped = False
        theta = 0

    info = {
        'v0': stim._v.tolist(),
        'v1': stim.vertices.tolist(),
        'flipped': flipped,
        'theta': theta,
        'stimulus': stim_path.namebase.split("_")[0],
    }

    return info


def save_configs(exp, force=False):
    example_path = STIM_PATH.joinpath("%s-example" % exp)
    training_path = STIM_PATH.joinpath("%s-training" % exp)
    exp_path = STIM_PATH.joinpath(exp)

    examples = map(get_stiminfo, example_path.listdir())
    training = map(get_stiminfo, training_path.listdir())
    exp_stims = {stim.name: get_stiminfo(stim) for stim in exp_path.listdir()}

    stim_pairs = pd.DataFrame.from_dict(exp_stims).T

    stims = stim_pairs['stimulus']\
        .drop_duplicates()
    stims = sorted(stims.tolist()) * 10

    # overrepresent theta=0 and theta=180 so they are shown the same
    # number of times as the other angles
    thetas = stim_pairs[['theta', 'flipped']]\
        .drop_duplicates()\
        .sort(['theta', 'flipped'])
    theta0 = thetas.ix[thetas['theta'] == 0]
    theta180 = thetas.ix[thetas['theta'] == 180]
    thetas = thetas.append(theta0).append(theta180)
    flipped = thetas['flipped'].tolist() * 5
    thetas = thetas['theta'].tolist() * 5

    trials = []
    for i in xrange(4):
        stims_i = stims
        if i > 0:
            flipped_i = flipped[-i:] + flipped[:-i]
            thetas_i = thetas[-i:] + thetas[:-i]
        else:
            flipped_i = flipped
            thetas_i = thetas

        full_block = zip(stims_i, thetas_i, flipped_i)
        if len(full_block) % 2 != 0:
            raise ValueError("blocks are not divisible by two")

        n = len(full_block) / 2
        trials.append(full_block[:n])
        trials.append(full_block[n:])

    stim_pairs = stim_pairs.set_index(['stimulus', 'theta', 'flipped'])
    for i, block in enumerate(trials):
        t = stim_pairs.ix[trials[i]].reset_index()

        config = {}
        config['training'] = training
        config['experiment'] = sorted(t.T.to_dict().values())
        config['examples'] = examples

        config_path = EXP_PATH.joinpath(
            "static", "json", "%s-cb0.json" % i).abspath()

        if config_path.exists() and not force:
            continue

        if not config_path.dirname().exists():
            config_path.dirname().makedirs_p()

        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2, allow_nan=False)

        logger.info("Saved %s", config_path.relpath())


def make_parser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force configs to be generated.")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    save_configs(args.exp, force=args.force)
