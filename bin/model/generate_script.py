#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation.sims.build import build
from copy import deepcopy

MODELS = ['GoldStandardModel', 'HillClimbingModel', 'BayesianQuadratureModel']

def make_parser():
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
        help="Experiment version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force script to be generated.")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    params = {}
    params['model'] = args.model
    params['exp'] = args.exp
    params['force'] = args.force
    params['stim_path'] = args.exp
    build(**params)
