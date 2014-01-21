#!/usr/bin/env python

import argparse
import multiprocessing as mp
from mental_rotation import MODELS
from mental_rotation.sims.server import run_server
from mental_rotation.sims.client import run_client
from ConfigParser import SafeConfigParser
from path import path

# load configuration
config = SafeConfigParser()
config.read("config.ini")


def get_params(model, version, sim_path, script_path):
    """Load the parameters from the simulation script."""
    return script


def parse_and_run_server(parser_args):
    SIM_PATH = path(config.get("paths", "simulations"))
    SCRIPT_PATH = path(config.get("paths", "sim_scripts"))

    model = parser_args.model
    version = parser_args.version
    sim_root = SIM_PATH.joinpath(model, version)
    script_root = SCRIPT_PATH.joinpath(model)
    script_file = script_root.joinpath("%s.json" % version)
    with script_file.open("r") as fid:
        script = json.load(fid)
    script["script_root"] = str(script_root)
    script["sim_root"] = str(sim_root)
    script["tasks_path"] = str(sim_root.joinpath("tasks.json"))

    kwargs = dict(parser_args._get_kwargs())
    del kwargs['model']
    del kwargs['version']
    del kwargs['func']
    run_server(script, **kwargs)


def parse_and_run_client(parser_args):
    kwargs = dict(parser_args._get_kwargs())
    del kwargs['func']
    run_client(**kwargs)


def create_server_parser(parser):
    VERSION = config.get("global", "version")

    def parse_address(address):
        host, port = address.split(":")
        return host, int(port)

    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=MODELS,
        help="Model name.")
    parser.add_argument(
        "-v", "--version",
        default=VERSION,
        help="Experiment/code version.")
    parser.add_argument(
        "-a", "--address",
        default=("127.0.0.1", 50000),
        type=parse_address,
        help="Address (host:port) of server.")
    parser.add_argument(
        "-k", "--authkey",
        default=None,
        help="Server authentication key.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force all tasks to be put on the queue.")
    parser.add_argument(
        "-l", "--loglevel",
        default="INFO",
        dest="loglevel",
        type=int,
        help="Logging verbosity level.")
    parser.set_defaults(func=parse_and_run_server)


def create_client_parser(parser):
    def parse_address(address):
        host, port = address.split(":")
        return host, int(port)

    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save data to disk.")
    parser.add_argument(
        "-T", "--timeout",
        default=310,
        type=int,
        help="Timeout (in seconds) before process restarts.")
    parser.add_argument(
        "-a", "--address",
        default=("127.0.0.1", 50000),
        type=parse_address,
        help="Address (host:port) of server.")
    parser.add_argument(
        "-k", "--authkey",
        default=None,
        help="Server authentication key.")
    parser.add_argument(
        "-n", "--num-processes",
        default=mp.cpu_count(),
        dest="num_procs",
        type=int,
        help="Number of client processes.")
    parser.add_argument(
        "-m", "--max-tries",
        default=3,
        dest="max_tries",
        type=int,
        help="Number of times to try running a task.")
    parser.add_argument(
        "-l", "--loglevel",
        default="INFO",
        dest="loglevel",
        type=str,
        help="Logging verbosity level.")

    parser.set_defaults(func=parse_and_run_client)



if __name__ == "__main__":
    # create main parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()

    # create server parser
    server = subparsers.add_parser(
        "server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_server_parser(server)

    # create client parser
    client = subparsers.add_parser(
        "client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_client_parser(client)

    args = parser.parse_args()
    args.func(args)
