#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation import MODELS
from mental_rotation.sims.manager import run
from ConfigParser import SafeConfigParser, NoOptionError
from path import path


def make_params(model, config):
    version = config.get("global", "version")

    def parse(vals):
        return [float(x.strip()) for x in vals.split(",")]

    model_opts = {
        'S_sigma': parse(config.get("sims", "s_sigma")),
        'step': parse(config.get("sims", "step")),
        'prior': parse(config.get("sims", "prior"))
    }

    sim_path = path(config.get("paths", "simulations"))
    sim_root = path(sim_path.joinpath(model, version))
    stim_path = path(config.get("paths", "stimuli")).joinpath(version)

    try:
        stim_ids = [x.strip() for x in config.get("sims", "stims").split(",")]
    except NoOptionError:
        stim_ids = sorted(set([
            x.namebase.split("_")[0] for x in stim_path.listdir()]))

    stim_paths = []
    for stim in stim_path.listdir():
        if stim.namebase.split("_")[0] in stim_ids:
            stim_paths.append(stim)

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
        default='127.0.0.1',
        help="Server hostname")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=55556,
        help="Server port number")
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
