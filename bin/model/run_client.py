#!/usr/bin/env python

import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser

from mental_rotation.sims.client import run

if __name__ == "__main__":

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-H", "--host",
        default='127.0.0.1',
        help="server host")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=55556,
        help="server port")
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=mp.cpu_count(),
        help="number of processes to use")
    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)
    loglevel = config.get("global", "loglevel")

    run(args.host, args.port, args.num, loglevel)
