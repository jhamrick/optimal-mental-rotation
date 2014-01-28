#!/usr/bin/python

import sys
import subprocess

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from mental_rotation import MODELS
from termcolor import colored
from path import path


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=MODELS,
        help="Name of the model to use.")
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
    version = config.get("global", "version")

    sim_path = path(config.get("paths", "simulations"))
    sim_root = path(sim_path.joinpath(model, version))

    zip_path = sim_root.dirname().joinpath(version + ".tar.gz")
    if zip_path.exists() and not args.force:
        print "'%s' already exists" % zip_path.relpath()
        sys.exit(0)

    if zip_path.exists():
        zip_path.remove()

    cmd = ["tar", "-czvf", zip_path, '-C', sim_root.dirname(), sim_root.name]
    print colored(" ".join(cmd), 'blue')
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError("tar exited with code %d" % code)
