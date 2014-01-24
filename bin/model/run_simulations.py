#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation import MODELS
from mental_rotation.sims.manager import run
from ConfigParser import SafeConfigParser
from path import path


def make_params(model, config):
    version = config.get("global", "version")
    model_opts = {
        'S_sigma': config.getfloat("model", "S_sigma")
    }

    sim_path = path(config.get("paths", "simulations"))
    sim_root = path(sim_path.joinpath(model, version))
    stim_path = path(config.get("paths", "stimuli"))
    stim_paths = stim_path.joinpath(version).listdir()

    params = {
        'model': model,
        'version': version,
        'num_samples': config.getint(model, 'num_samples'),
        'chunksize': config.getint(model, 'chunksize'),
        'sim_root': str(sim_root),
        'tasks_path': str(sim_root.joinpath("tasks.json")),
        'completed_path': str(sim_root.joinpath("completed.json")),
        'stim_paths': map(str, stim_paths),
        'model_opts': model_opts,
        'loglevel': config.get("global", "loglevel")
    }

    return params


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=MODELS,
        help="Name of the model to use.")
    parser.add_argument(
        "-H", "--host",
        default='127.0.0.1')
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=55556)
    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force script to be generated.")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)
    model = args.model

    params = make_params(model, config)
    run(args.host, args.port, params, args.force)
