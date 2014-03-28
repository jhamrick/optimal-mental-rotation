#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from itertools import product
from mental_rotation.stimulus import Stimulus2D
from path import path
import json
import logging
import pandas as pd

logger = logging.getLogger("experiment.generate_configs")


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


def save_configs(version, stim_path, exp_path, force):
    example_path = stim_path.joinpath("%s-example" % version)
    training_path = stim_path.joinpath("%s-training" % version)
    stim_path = stim_path.joinpath(version)

    examples = map(get_stiminfo, example_path.listdir())
    training = map(get_stiminfo, training_path.listdir())
    exp_stims = {stim.name: get_stiminfo(stim) for stim in stim_path.listdir()}

    stim_pairs = pd.DataFrame.from_dict(exp_stims).T
    stim_pairs['theta'] = stim_pairs['theta'].astype(int)

    stims = sorted(
        stim_pairs['stimulus']
        .drop_duplicates()
        .tolist())

    # overrepresent theta=0 and theta=180 so they are shown the same
    # number of times as the other angles
    thetas = sorted((
        stim_pairs['theta']
        .drop_duplicates()
        .tolist() + [-180, 360]))

    # combine the stimuli and rotations
    items = []
    for i in xrange(20):
        s = stims[:]
        if i == 0:
            t = thetas[:]
        else:
            t = thetas[-i:] + thetas[:-i]
        items.extend([[s[i], t[i]] for i in xrange(len(s))])

    # now add in flip/same and separate into blocks
    trials = []
    for i in xrange(8):
        st = items[(i * 50):((i + 1) * 50)]
        block = []
        block.extend([x + [True] for x in st])
        block.extend([x + [False] for x in st])
        trials.append(block)

    # sanity check, to make sure we got all the combinations
    all_trials = reduce(lambda x, y: x + y, trials, [])
    assert len(all_trials) == 800
    all_trials = pd.DataFrame(sorted(all_trials)).drop_duplicates()
    assert len(all_trials) == 800

    # go back through and rename -180 to 180 and 360 to 0
    for block in trials:
        for i in xrange(len(block)):
            if block[i][1] == -180:
                block[i][1] = 180
            elif block[i][1] == 360:
                block[i][1] = 0
            block[i] = tuple(block[i])

    stim_pairs = stim_pairs.set_index(['stimulus', 'theta', 'flipped'])
    for i, block in enumerate(trials):
        tA = stim_pairs.ix[trials[i]].reset_index()
        if i < (len(trials) - 1):
            tB = stim_pairs.ix[trials[i+1]].reset_index()
        else:
            tB = stim_pairs.ix[trials[0]].reset_index()

        trial_config = {}
        trial_config['training'] = training
        trial_config['experimentA'] = sorted(tA.T.to_dict().values())
        trial_config['experimentB'] = sorted(tB.T.to_dict().values())
        trial_config['examples'] = examples

        trial_config_path = exp_path.joinpath(
            "static", "json", "%s-cb0.json" % i).abspath()

        if trial_config_path.exists() and not force:
            continue

        if not trial_config_path.dirname().exists():
            trial_config_path.dirname().makedirs_p()

        with open(trial_config_path, "w") as fh:
            json.dump(trial_config, fh, indent=2, allow_nan=False)

        logger.info("Saved %s", trial_config_path.relpath())


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force configs to be generated.")

    args = parser.parse_args()
    config = SafeConfigParser()
    config.read(args.config)

    loglevel = config.get("global", "loglevel")
    logging.basicConfig(level=loglevel)

    version = config.get("global", "version")
    stim_path = path(config.get("paths", "stimuli"))
    exp_path = path(config.get("paths", "experiment"))

    save_configs(version, stim_path, exp_path, args.force)
