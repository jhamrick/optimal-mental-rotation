#!/usr/bin/env python

import argparse
from mental_rotation.sims.server import create_server_parser
from mental_rotation.sims.client import create_client_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()

    server = subparsers.add_parser(
        "server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_server_parser(server)

    client = subparsers.add_parser(
        "client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_client_parser(client)

    args = parser.parse_args()
    args.func(args)
