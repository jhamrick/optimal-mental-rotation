#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser

from mental_rotation.sims.client import run

if __name__ == "__main__":

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-H", "--host",
        required=True)
    parser.add_argument(
        "-p", "--port",
        type=int,
        required=True)
    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)
    loglevel = config.get("global", "loglevel")

    run(args.host, args.port, loglevel)
