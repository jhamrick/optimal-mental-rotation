#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import numpy as np
from mental_rotation import EXP_PATH, STIM_PATH
from mental_rotation.stimulus import Stimulus2D
import logging

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
    all_stims = STIM_PATH.joinpath(exp).listdir()
    all_stims = [x for x in all_stims if x.name.split("_")[0] != "example"]

    stim_pairs = np.array([get_stiminfo(stim) for stim in all_stims])
    examples = [
        get_stiminfo(STIM_PATH.joinpath(exp, "example_320_0.json")),
        get_stiminfo(STIM_PATH.joinpath(exp, "example_320_1.json"))
    ]

    stim_pairs = np.array(stim_pairs)
    np.random.shuffle(stim_pairs)
    conds = stim_pairs.reshape((10, -1))

    for i, cond in enumerate(conds):
        config = {}
        config['trials'] = cond.tolist()
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