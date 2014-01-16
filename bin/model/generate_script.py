#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mental_rotation.sims.build import build
from mental_rotation import MODELS
from copy import deepcopy


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


def make_params(model, exp, force):
    params = {
        'model': model,
        'exp': exp,
        'force': force,
        'stim_path': exp
    }

    if model == "GoldStandardModel":
        params['num_samples'] = 1
        params['chunksize'] = 10
    elif model == "HillClimbingModel":
        params['num_samples'] = 100
        params['chunksize'] = 100
    elif model == "BayesianQuadratureModel":
        params['num_samples'] = 10
        params['chunksize'] = 2
    else:
        raise ValueError("unhandled model type: %s" % model)

    return params


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    params = make_params(args.model, args.exp, args.force)
    build(**params)
