#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation.sims.build import build
from mental_rotation import MODELS
from copy import deepcopy
from ConfigParser import SafeConfigParser
from path import path


# load configuration
config = SafeConfigParser()
config.read("config.ini")


def make_parser():
    VERSION = config.get("global", "version")

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=MODELS,
        help="Name of the model to use.")
    parser.add_argument(
        "-v", "--version",
        default=VERSION,
        help="Experiment/code version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force script to be generated.")

    return parser


def make_params(model, version, force):
    params = {
        'model': model,
        'version': version,
        'force': force,
        'stim_path': version
    }

    if model == "GoldStandardModel":
        params['num_samples'] = 1
        params['chunksize'] = 1
    elif model == "HillClimbingModel":
        params['num_samples'] = 100
        params['chunksize'] = 100
    elif model == "ThresholdModel":
        params['num_samples'] = 100
        params['chunksize'] = 100
    elif model == "BayesianQuadratureModel":
        params['num_samples'] = 10
        params['chunksize'] = 2
    else:
        raise ValueError("unhandled model type: %s" % model)

    return params


def build(model, version, **params):
    """Create a simulation script."""

    SIM_PATH = path(config.get("paths", "simulations"))
    SCRIPT_PATH = path(config.get("paths", "sim_scripts"))
    STIM_PATH = path(config.get("paths", "stimuli"))

    force = params['force']

    # Path where we will save the simulations
    sim_root = SIM_PATH.joinpath(model, version)

    # Path where we will save the simulation script/resources
    script_root = SCRIPT_PATH.joinpath(model)
    script_file = script_root.joinpath("%s.json" % version)

    # check to see if we would override existing data
    if not force and script_file.exists():
        logger.debug("Script %s already exists", script_file.relpath())
        return

    # remove existing files, if we're overwriting
    if script_root.exists():
        script_root.rmtree()

    # Locations of stimuli
    stim_paths = STIM_PATH.joinpath(params['stim_path']).listdir()

    # Put it all together in a big dictionary...
    script = {}

    # Simulation version
    script['model'] = model
    script['version'] = version

    # Various paths -- but strip away the absolute parts, because we
    # might be running the simulations on another computer
    script['sim_root'] = str(sim_root.relpath(SIM_PATH))
    script['stim_paths'] = [str(x.relpath(STIM_PATH)) for x in stim_paths]

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
    parser = make_parser()
    args = parser.parse_args()

    params = make_params(args.model, args.version, args.force)
    build(**params)
