#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from termcolor import colored
from path import path
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
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")

    group = parser.add_argument_group(title="post-experiment operations")
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="perform all post-experiment actions")
    group.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="fetch data from experiment server")
    group.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="process raw data into datapackages")
    group.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="extract worker ids")

    args = parser.parse_args()
    required = [
        args.fetch,
        args.process,
        args.extract,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    config = args.config
    force = args.force

    # download data
    if args.fetch or args.all:
        cmd = [
            "python", "./bin/experiment/fetch_data.py",
            "-c", config
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # process data
    if args.process or args.all:
        cmd = [
            "python", "./bin/experiment/process_data.py",
            "-c", config
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # extract ids
    if args.extract or args.all:
        cmd = [
            "python", "./bin/experiment/extract_workers.py",
            "-c", config
        ]
        run_cmd(cmd)
