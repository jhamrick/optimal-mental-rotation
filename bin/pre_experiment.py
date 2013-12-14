#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation import BIN_PATH
from termcolor import colored
import logging
import subprocess
import sys

logger = logging.getLogger('mental_rotation.experiment')


def run_cmd(cmd):
    logging.info(colored("Running %s" % " ".join(cmd), 'blue'))
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError("Process exited abnormally: %d" % code)


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="experiment version")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")

    group = parser.add_argument_group(title="pre-experiment operations")
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="perform all pre-experiment actions")
    group.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="generate trial configurations")
    group.add_argument(
        "--deploy",
        action="store_true",
        default=False,
        help="deploy experiment files to server")

    args = parser.parse_args()
    required = [
        args.generate,
        args.deploy,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    exp = args.exp
    force = args.force

    # generate configs
    if args.generate or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("experiment/generate_configs.py"),
            "-e", exp
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # deploy experiment files
    if args.deploy or args.all:
        run_cmd([
            "python", BIN_PATH.joinpath("experiment/deploy_experiment.py")
        ])
