#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from termcolor import colored
from mental_rotation import MODELS

import logging
import subprocess
import sys


def run_cmd(cmd):
    logging.info(colored("Running %s" % " ".join(cmd), 'blue'))
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError("Process exited abnormally: %d" % code)


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
        help="force tasks to complete")

    group = parser.add_argument_group(title="simulate operations")
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="perform all actions")
    group.add_argument(
        "--run",
        dest="run",
        action="store_true",
        default=False,
        help="run simulations")
    group.add_argument(
        "--zip",
        action="store_true",
        default=False,
        help="create a gzip archive of simulation data")
    group.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="process simulation data")

    args = parser.parse_args()
    required = [
        args.run,
        args.zip,
        args.process,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    model = args.model
    config = args.config
    force = args.force

    # run simulations
    if args.run or args.all:
        cmd = [
            "python", "./bin/model/run_simulations.py",
            "-m", model, "-c", config
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # zip simulations
    if args.zip or args.all:
        cmd = [
            "python", "./bin/model/zip_data.py",
            "-m", model, "-c", config
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # process simulation data
    if args.process or args.all:
        cmd = [
            "python", "./bin/model/process_simulations.py",
            "-m", model, "-c", config
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)
