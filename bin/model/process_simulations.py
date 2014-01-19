#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import path
import json
import logging
import pandas as pd

from mental_rotation import SIM_PATH, DATA_PATH, MODELS
from mental_rotation import model as m
from snippets import datapackage as dpkg

logger = logging.getLogger('mental_rotation')


def load(model_class, pth, stimname):
    model = model_class.load(pth)
    stim, rot, flip = stimname.split("_")
    rot = float(rot)

    hypotheses = {
        0: "same",
        1: "flipped"
        None: None
    }
    flip = hypotheses[int(flip)]

    data = {}
    data['nstep'] = model.R_i.size
    data['hypothesis'] = hypotheses[model.hypothesis_test()]
    data['log_lh_h0'] = model.log_lh_h0
    data['log_lh_h1'] = model.log_lh_h1
    data['stimulus'] = stim
    data['theta'] = rot
    data['flipped'] = flip

    return data

def process_all(model_type, exp, force=False):
    name = "%s_%s.dpkg" % (model_type, exp)
    dp_path = DATA_PATH.joinpath("model", name)

    if dp_path.exists() and not force:
        return

    sim_root = SIM_PATH.joinpath(model_type, exp)
    tasks_file = sim_root.joinpath("tasks.json")
    with open(tasks_file, "r") as fh:
        tasks = json.load(fh)
    completed_file = sim_root.joinpath("completed.json")
    with open(completed_file, "r") as fh:
        completed = json.load(fh)
        
    try:
        model_class = getattr(m, model_type)
    except AttributeError:
        raise ValueError("unhandled model type: %s" % model_type)

    data = []
    for i, taskname in enumerate(sorted(tasks.keys())):
        task = tasks[taskname]
        pth = path(task['data_path'])
        logger.info("Processing '%s'...", taskname)

        if not completed[taskname]:
            raise RuntimeError("simulations are not complete")

        for isample in task['samples']:
            samppth = pth.joinpath("sample_%d" % isample)
            sampdata = load(model_class, samppth, pth.namebase)
            sampdata['sample'] = isample
            data.append(sampdata)

    df = pd.DataFrame(data)

    # load the existing datapackage and bump the version
    if dp_path.exists():
        dp = dpkg.DataPackage.load(dp_path)
        dp.bump_minor_version()
        dp.clear_resources()

    # create the datapackage
    else:
        dp = dpkg.DataPackage(name=name, licenses=['odc-by'])
        dp['version'] = '1.0.0'
        dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
        dp.add_contributor("Thomas L. Griffiths", "tom_griffiths@berkeley.edu")

    dp.add_resource(dpkg.Resource(
        name="model.csv", fmt="csv",
        data=df, pth="./model.csv"))

    # save
    dp.save(dp_path.dirname())
    logger.info("Saved to '%s'" % dp_path.relpath())


if __name__ == "__main__":
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
        help="Force processed data to be overwritten.")

    args = parser.parse_args()
    process_all(args.model, args.exp, force=args.force)
