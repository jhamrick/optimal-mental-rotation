#!/usr/bin/env python

import argparse
import dbtools
import logging
import pandas as pd
from ConfigParser import SafeConfigParser
from snippets import datapackage as dpkg
from path import path

logger = logging.getLogger("mental_rotation.experiment")

# load configuration
config = SafeConfigParser()
config.read("config.ini")


def get_table():
    DATA_PATH = path(config.get("paths", "data"))
    dbpath = DATA_PATH.joinpath("human", "workers.db")

    if not dbtools.Table.exists(dbpath, "workers"):
        logger.info("Creating new table 'workers'")
        tbl = dbtools.Table.create(
            dbpath, "workers",
            [('pid', str),
             ('dataset', str)])

    else:
        logger.info("Loading existing table 'workers'")
        tbl = dbtools.Table(dbpath, "workers")

    return tbl


def save_stability(version):
    DATA_PATH = path(config.get("paths", "data"))
    dp_path = DATA_PATH.joinpath("human", "%s.dpkg" % version)

    # load the datapackage
    logger.info("Loading '%s'", dp_path.relpath())
    dp = dpkg.DataPackage.load(dp_path)

    # is there a turk.csv file?
    try:
        turk = dp.load_resource("turk.csv")
    except KeyError:
        turk = None

    # is there a participants.csv file?
    try:
        parts = dp.load_resource("participants.csv")
    except KeyError:
        parts = None

    # get the worker ids
    if turk:
        workers = sorted(turk.reset_index()['WorkerId'].unique())
    elif parts:
        workers = sorted(parts['pid'].unique())
    else:
        logger.warning("'%s' is not a Mechanical Turk experiment", version)
        return

    # create a new dataframe
    df = pd.DataFrame({
        'pid': workers
    })
    df['dataset'] = str(dp_path.name)

    # load the table we're saving it to
    tbl = get_table()

    KEY = ['pid']
    tbl_dupes = tbl.select(KEY, where=("dataset=?", dp_path.name))
    tbl_dupes['pid'] = map(str, tbl_dupes['pid'])
    tbl_dupes = set([x[0] for x in tbl_dupes.to_records(index=False).tolist()])
    df_dupes = set([x[0] for x in df[KEY].to_records(index=False).tolist()])

    # get the unique values and the duplicated values, because we will
    # treat them differently
    unique = pd.Index(df_dupes.difference(tbl_dupes), name="pid")
    df_idx = df.set_index(KEY)

    if len(unique) > 0:
        logger.info("Adding %d new items", len(unique))
        tbl.insert(df_idx.ix[unique].reset_index().T.to_dict().values())

if __name__ == "__main__":
    VERSION = config.get("global", "version")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-v", "--version",
        default=VERSION,
        help="Experiment version.")

    args = parser.parse_args()
    save_stability(args.version)
