#!/usr/bin/env python

import argparse
import dbtools
import logging
import pandas as pd
from ConfigParser import SafeConfigParser
from snippets import datapackage as dpkg
from path import path

logger = logging.getLogger("experiment.extract_workers")


def get_table(data_path):
    dbpath = data_path.joinpath("human", "workers.db")

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


def extract_workers(version, data_path):
    dp_path = data_path.joinpath("human", "%s.dpkg" % version)

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
    tbl = get_table(data_path)

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")

    args = parser.parse_args()

    config = SafeConfigParser()
    config.read(args.config)

    loglevel = config.get("global", "loglevel")
    logging.basicConfig(level=loglevel)

    version = config.get("global", "version")
    data_path = path(config.get("paths", "data"))

    extract_workers(version, data_path)
