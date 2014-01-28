#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import path
from ConfigParser import SafeConfigParser
import json
import logging
import pandas as pd
import multiprocessing as mp

from mental_rotation import MODELS
from mental_rotation import model as m
from snippets import datapackage as dpkg

logger = logging.getLogger('model.process_simulations')


def load(task):
    model_opts = task['model_opts']
    model_class = getattr(m, task['model'])
    data_path = path(task['data_path'])
    stim_path = path(task['stim_path'])

    all_data = []

    for iopt, opts in model_opts.iteritems():
        samppth = data_path.joinpath("part_%s" % iopt)
        stimname = stim_path.namebase

        model = model_class.load(samppth)
        stim, rot, flip = stimname.split("_")
        rot = float(rot)

        hypotheses = {
            0: "same",
            1: "flipped",
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
        data.update(opts)
        all_data.append(data)

    return task['task_name'], all_data


def process_all(model_type, version, sim_path, data_path, force=False):
    name = "%s_%s.dpkg" % (model_type, version)
    dp_path = data_path.joinpath("model", name)

    if dp_path.exists() and not force:
        return

    sim_root = sim_path.joinpath(model_type, version)
    tasks_file = sim_root.joinpath("tasks.json")
    with open(tasks_file, "r") as fh:
        tasks = json.load(fh)
    completed_file = sim_root.joinpath("completed.json")
    with open(completed_file, "r") as fh:
        completed = json.load(fh)

    for taskname in tasks:
        if not completed[taskname]:
            raise RuntimeError("task '%s' is not complete" % taskname)

    pool = mp.Pool()
    results = pool.imap_unordered(load, tasks.values())

    data = []
    for i, result in enumerate(results):
        taskname, taskdata = result
        logger.info(
            "[%d/%d] Successfully processed '%s'",
            i + 1, len(tasks), taskname)
        data.extend(taskdata)

    df = pd.DataFrame(data)\
           .sort(['stimulus', 'theta', 'flipped', 'sample'])\
           .reset_index(drop=True)

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
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force processed data to be overwritten.")

    args = parser.parse_args()

    config = SafeConfigParser()
    config.read(args.config)
    sim_path = path(config.get("paths", "simulations"))
    data_path = path(config.get("paths", "data"))
    version = config.get("global", "version")
    loglevel = config.get("global", "loglevel")
    logging.basicConfig(level=loglevel)
    logger.setLevel(loglevel)

    process_all(args.model, version, sim_path, data_path, force=args.force)
