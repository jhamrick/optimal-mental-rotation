#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from termcolor import colored
from path import path
import subprocess

# load configuration
config = SafeConfigParser()
config.read("config.ini")


if __name__ == "__main__":
    EXP_PATH = path(config.get("paths", "experiment"))

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-u", "--user",
        default="cocosci",
        help="Username to login to the server.")
    parser.add_argument(
        "-H", "--host",
        default="cocosci.berkeley.edu",
        help="Hostname of the experiment server.")
    parser.add_argument(
        "-n", "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Show what would have been transferred.")
    parser.add_argument(
        "--bwlimit",
        type=int,
        default=None,
        help="Bandwidth limit for transfer")
    parser.add_argument(
        "dest",
        default="cocosci-python.dreamhosters.com/experiment/",
        nargs="?",
        help="Destination path on the experiment server.")

    args = parser.parse_args()

    src_paths = [
        str(EXP_PATH.joinpath("static").relpath()),
        str(EXP_PATH.joinpath("templates").relpath())
    ]

    cmd_template = ["rsync", "-av", "--delete-after"]
    if args.dry_run:
        cmd_template.append("-n")
    if args.bwlimit:
        cmd_template.append("--bwlimit=%d" % args.bwlimit)
    cmd_template.append("%s")
    cmd_template.append("%s")

    dest = "%s@%s:%s" % (args.user, args.host, args.dest)
    for source in src_paths:
        cmd = " ".join(cmd_template) % (source, dest)
        print colored(cmd, 'blue')
        code = subprocess.call(cmd, shell=True)
        if code != 0:
            raise RuntimeError("rsync exited abnormally: %d" % code)
