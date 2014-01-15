#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation import BIN_PATH, MODELS
from termcolor import colored
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
        "-e", "--exp",
        required=True,
        help="experiment version")
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
        "--generate",
        action="store_true",
        default=False,
        help="generate simulation scripts")
    group.add_argument(
        "--run-server",
        dest="run_server",
        action="store_true",
        default=False,
        help="run simulation server")
    group.add_argument(
        "--run-client",
        dest="run_client",
        action="store_true",
        default=False,
        help="run simulation client")
    group.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="process simulation data")

    args = parser.parse_args()
    required = [
        args.generate,
        args.run_server,
        args.run_client,
        args.process,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    model = args.model
    exp = args.exp
    force = args.force

    # generate configs
    if args.generate or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("model/generate_script.py"),
            "-m", model, "-e", exp
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # run experiment server
    if args.run_server or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("model/run_simulations.py"),
            "server", "-m", model, "-e", exp, "-k", "zo7MV6GndfNf"
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # run experiment client
    if args.run_client:
        run_cmd([
            "python", BIN_PATH.joinpath("model/run_simulations.py"),
            "client", "-k", "zo7MV6GndfNf", "-s"
        ])

    # process simulation data
    if args.process or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("model/process_simulations.py"),
            "-m", model, "-e", exp
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)
