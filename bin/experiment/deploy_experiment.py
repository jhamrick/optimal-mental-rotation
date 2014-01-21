#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from termcolor import colored
from path import path
import subprocess


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)

    EXP_PATH = path(config.get("paths", "experiment"))
    src_paths = [
        str(EXP_PATH.joinpath("static").relpath()),
        str(EXP_PATH.joinpath("templates").relpath())
    ]

    cmd_template = ["rsync", "-av", "--delete-after"]
    cmd_template.append("%s")
    cmd_template.append("%s")

    dest = config.get("experiment", "remote_dest")
    for source in src_paths:
        cmd = " ".join(cmd_template) % (source, dest)
        print colored(cmd, 'blue')
        code = subprocess.call(cmd, shell=True)
        if code != 0:
            raise RuntimeError("rsync exited abnormally: %d" % code)
