#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from xmlrpclib import ServerProxy


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

    args = parser.parse_args()

    pandaserver = ServerProxy("http://%s:%d" % (args.host, args.port))
    pandaserver.panda_reload()
