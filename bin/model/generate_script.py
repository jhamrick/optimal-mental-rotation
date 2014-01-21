#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation.sims.build import build
from mental_rotation import MODELS
from copy import deepcopy
from ConfigParser import SafeConfigParser
from path import path
import logging

logger = logging.getLogger("model.generate_script")


def make_params(model, version, force):

    return params


def build(model, version, **params):
    """Create a simulation script."""

    sim_path = params['sim_path']
    script_path = params['script_path']
    stim_path = params['stim_path']
    force = params['force']

    # Path where we will save the simulations
    sim_root = sim_path.joinpath(model, version)

    # Path where we will save the simulation script/resources
    script_root = script_path.joinpath(model)
    script_file = script_root.joinpath("%s.json" % version)

    # check to see if we would override existing data
    if not force and script_file.exists():
        logger.debug("Script %s already exists", script_file.relpath())
        return

    # remove existing files, if we're overwriting
    if script_root.exists():
        script_root.rmtree()

    # Locations of stimuli
    stim_paths = stim_path.joinpath(version).listdir()

    # Put it all together in a big dictionary...
    script = {}

    # Simulation version
    script['model'] = model
    script['version'] = version

    # Various paths -- but strip away the absolute parts, because we
    # might be running the simulations on another computer
    script['sim_root'] = str(sim_root.relpath(sim_path))
    script['stim_paths'] = [str(x.relpath(stim_path)) for x in stim_paths]

    # number of samples
    script['num_samples'] = params['num_samples']
    script['chunksize'] = params['chunksize']

    # create the directory for our script and resources, and save them
    if not script_root.exists():
        script_root.makedirs_p()

    with script_file.open("w") as fh:
        json.dump(script, fh, indent=2)

    logger.info("Saved script to %s", script_file.relpath())

    return script


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=MODELS,
        help="Name of the model to use.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force script to be generated.")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)

    loglevel = config.get("global", "loglevel")
    logging.basicConfig(level=loglevel)

    params = {
        'model': args.model,
        'version': config.get("global", "version"),
        'force': args.force,
        'num_samples': config.getint(model, 'num_samples'),
        'chunksize': config.getint(model, 'chunksize')
        'stim_path': path(config.get("paths", "stimuli")),
        'sim_path': path(config.get("paths", "simulations")),
        'script_path': path(config.get("paths", "sim_scripts"))
    }

    build(**params)
