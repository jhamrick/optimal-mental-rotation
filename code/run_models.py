#!/usr/local/bin/python

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mental rotation models.')
    parser.add_argument('stims', metavar='stim', type=str, nargs='*',
                        help='name of stimulus to simulate')
    parser.add_argument('--gs', dest='model_gs', action='store_true',
                        default=False, help="Gold standard model")
    parser.add_argument('--naive', dest='model_naive', action='store_true',
                        default=False, help="Naive model")
    parser.add_argument('--vm', dest='model_vm', action='store_true',
                        default=False, help="Von Mises model")
    parser.add_argument('--bq', dest='model_bq', action='store_true',
                        default=False, help="Bayesian Quadrature model")

    args = parser.parse_args()
    run_models = {
        'GoldStandard': args.model_gs,
        'Naive': args.model_naive,
        'VonMises': args.model_vm,
        'BayesianQuadrature': args.model_bq
    }

    if not any(run_models.values()):
        run_models = dict((key, True) for key in run_models)

    import util
    import models

    opt = util.load_opt()
    for mname, run in run_models.iteritems():
        model = getattr(models, mname)
        util.run_all(args.stims, model, opt)
