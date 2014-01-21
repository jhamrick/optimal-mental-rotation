#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from mental_rotation.stimulus import Stimulus2D
from path import path
import logging
import numpy as np
import sys

logger = logging.getLogger('mental_rotation.experiment')


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


def convert_stims(from_path, to_path, force, config):
    STIM_PATH = path(config.get("paths", "stimuli"))
    stim_path = STIM_PATH.joinpath(from_path)
    dest = STIM_PATH.joinpath(to_path)

    if dest.exists() and not force:
        logger.warning(
            "Destination already exists, exiting (use --force to override)")
        return

    files = stim_path.listdir()
    files = [x for x in files if x.endswith(".npz")]

    if dest.exists():
        dest.rmtree_p()

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
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")

    args = parser.parse_args()

    config = SafeConfigParser()
    config.read(args.config)

    convert_stims(args.from_path, args.to_path, args.force, config)
