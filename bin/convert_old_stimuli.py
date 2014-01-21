#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from mental_rotation.stimulus import Stimulus2D
from path import path
import logging
import numpy as np
import sys

logger = logging.getLogger('mental_rotation.experiment')


# load configuration
config = SafeConfigParser()
config.read("config.ini")


def load_stim(file):
    logger.debug("Loading stimulus '%s'" % file.relpath())
    data = np.load(file)

    # stim1 vertices for each rotation
    Xm = data['Xm']
    assert (Xm[0, 0] == Xm[0, -1]).all()

    # stim2 vertices
    Xb = data['Xb']

    # we won't actually use the following variables -- but leaving
    # this block of code in for reference

    # rotations
    R = data['R']
    # image/noisy vertices
    Im = data['Im']
    Ib = data['Ib']
    # actual rotation of stim1
    theta = data['theta']

    data.close()

    # stimulus id, rotation between stim1 and stim2, hypothesis
    stimname, rot, hyp = file.splitext()[0].splitpath()[1].split("_")
    rot = float(rot)
    assert hyp in ('h0', 'h1')

    stim = Stimulus2D(Xm[0, :-1])
    if hyp == 'h0':
        stim.flip([0, 1])
        flipped = True
    else:
        flipped = False
    if rot != 0:
        stim.rotate(rot)
    assert np.allclose(stim.vertices, Xb[:-1])

    return (stimname, rot, flipped), stim


def convert_stims(stim_path, dest):
    files = stim_path.listdir()
    files = [x for x in files if x.endswith(".npz")]

    if not dest.exists():
        dest.makedirs_p()

    for f in files:
        (stimname, rot, flipped), stim = load_stim(f)
        filename = "%s_%d_%d.json" % (stimname, rot, flipped)
        stim.save(dest.joinpath(filename))

    logger.info("Saved stimuli to %s", dest.relpath())


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--from",
        dest="from_path",
        required=True,
        help="stimuli to convert")
    parser.add_argument(
        "--to",
        dest="to_path",
        required=True,
        help="new stimulus set name")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")

    args = parser.parse_args()
    STIM_PATH = path(config.get("paths", "stimuli"))
    from_path = STIM_PATH.joinpath(args.from_path)
    to_path = STIM_PATH.joinpath(args.to_path)

    stim_path = to_path.joinpath("stimuli.csv")
    trial_path = to_path.joinpath("trials.csv")

    if (stim_path.exists() or trial_path.exists()) and not args.force:
        logger.warning(
            "Destination already exists, exiting (use --force to override)")
        sys.exit(0)

    convert_stims(from_path, to_path)
