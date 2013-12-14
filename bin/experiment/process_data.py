#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from mental_rotation import DATA_PATH
from path import path
from snippets import datapackage as dpkg
import dbtools
import json
import logging
import numpy as np
import pandas as pd
import sys

logger = logging.getLogger('mental_rotation.experiment')


def str2bool(x):
    """Convert a string representation of a boolean (e.g. 'true' or
    'false') to an actual boolean.

    """
    sx = str(x)
    if sx.lower() == 'true':
        return True
    elif sx.lower() == 'false':
        return False
    else:
        return np.nan


def split_uniqueid(df, field):
    """PsiTurk outputs a field which is formatted as
    'workerid:assignmentid'. This function splits the field into two
    separate fields, 'pid' and 'assignment', and drops the old field
    from the dataframe.

    """

    workerid, assignmentid = zip(*map(lambda x: x.split(":"), df[field]))
    df['pid'] = workerid
    df['assignment'] = assignmentid
    df = df.drop([field], axis=1)
    return df


def parse_timestamp(df, field):
    """Parse JavaScript timestamps (which are in millseconds) to pandas
    datetime objects.

    """
    timestamp = pd.to_datetime(map(datetime.fromtimestamp, df[field] / 1e3))
    return timestamp


def find_bad_participants(exp, data):
    """Check participant data to make sure they pass the following
    conditions:

    1. No duplicated trials
    2. They finished the whole experiment

    Returns a dictionary of failed participants that includes the
    reasons why they failed.

    """

    participants = []
    for (assignment, pid), df in data.groupby(['assignment', 'pid']):
        info = {
            'pid': pid,
            'assignment': assignment,
            'note': None,
            'timestamp': None
        }

        # go ahead and add this to our list now -- the dictionary is
        # mutable, so when we update stuff later the dictionary in the
        # list will also be updated
        participants.append(info)

        # get the time they started the experiment
        times = df['psiturk_time'].copy()
        times.sort()
        start_time = pd.to_datetime(
            datetime.fromtimestamp(times.irow(0) / 1e3))
        info['timestamp'] = start_time

        # check for duplicated entries
        dupes = df.sort('psiturk_time')[['mode', 'trial', 'trial_phase']]\
                  .duplicated().any()
        if dupes:
            logger.warning("%s has duplicate trials", pid)
            info['note'] = "duplicate_trials"
            continue

        # check to make sure they actually finished
        prestim = df\
            .groupby(['trial_phase'])\
            .get_group('prestim')
        incomplete = len(prestim) != 100
        if incomplete:
            logger.warning("%s is incomplete", pid)
            info['note'] = "incomplete"
            continue

        # see if they already did (a version of) the experiment
        dbpath = DATA_PATH.joinpath("human", "workers.db")
        if dbtools.Table.exists(dbpath, "workers"):
            tbl = dbtools.Table(dbpath, "workers")
            datasets = tbl.select("dataset", where=("pid=?", pid))['dataset']
            exps = map(lambda x: path(x).namebase, datasets)
            if exp in exps:
                exps.remove(exp)
            if len(exps) > 0:
                logger.warning("%s is a repeat worker", pid)
                info['note'] = "repeat_worker"
                continue

        # check their accuracy
        exp_data = df.groupby('trial_phase')\
                     .get_group('stim')\
                     .groupby('mode')\
                     .get_group('experiment')
        accuracy = exp_data['flipped'] == exp_data['response']
        thresh = 0.25
        if accuracy.mean() <= (1 - thresh):
            logger.warning(
                "%s failed %d%% or more trials", pid, 100 * thresh)
            info['note'] = "failed"
            continue

    return participants


def load_meta(data_path):

    """Load experiment metadata from the given path. Returns a dictionary
    containing the metadata as well as a list of fields for the trial
    data.

    """
    # load the data and pivot it, so the rows are uniqueid, columns
    # are keys, and values are, well, values
    meta = pd.read_csv(data_path.joinpath(
        "questiondata_all.csv"), header=None)
    meta = meta.pivot(index=0, columns=1, values=2)

    # extract condition information for all participants
    conds = split_uniqueid(
        meta[['condition', 'counterbalance']].reset_index(),
        0).set_index('pid')
    conds['condition'] = conds['condition'].astype(int)
    conds['counterbalance'] = conds['counterbalance'].astype(int)
    conds['assignment'] = conds['assignment'].astype(str)
    conds = conds.T.to_dict()

    # make sure everyone saw the same questions/possible responses
    meta = meta.drop(['condition', 'counterbalance'], axis=1).drop_duplicates()
    assert len(meta) == 1

    # extract the field names
    fields = ["psiturk_id", "psiturk_currenttrial", "psiturk_time"]
    fields.extend(map(str, json.loads(meta['fields'][0])))

    # convert the remaining metadata to a dictionary and update it
    # with the parsed conditions
    meta = meta.drop(['fields'], axis=1).reset_index(drop=True).T.to_dict()[0]

    return meta, conds, fields


def load_data(data_path, conds, fields):
    """Load experiment trial data from the given path. Returns a pandas
    DataFrame.

    """
    # load the data
    data = pd.read_csv(data_path.joinpath(
        "trialdata_all.csv"), header=None)
    # set the column names
    data.columns = fields
    # split apart psiturk_id into pid and assignment
    data = split_uniqueid(data, 'psiturk_id')

    # process other various fields to make sure they're in the right
    # data format
    data['instructions'] = map(str2bool, data['instructions'])
    data['response_time'] = data['response_time'].astype('float') / 1e3
    data['flipped'] = map(str2bool, data['flipped'])
    data['flipped'] = data['flipped'].replace(
        {True: "flipped",
         False: "same"})
    data['theta'] = data['theta'].astype('float')
    data['response'] = data['response']
    data['trial_phase'] = data['trial_phase'].fillna('prestim')

    # remove instructions rows
    data = data\
        .groupby('instructions')\
        .get_group(False)

    # rename some columns
    data = data.rename(columns={
        'index': 'trial',
        'experiment_phase': 'mode'
    })
    # make trials be 1-indexed
    data['trial'] += 1

    # drop columns we don't care about
    data = data.drop([
        'psiturk_currenttrial',
        'instructions'], axis=1)

    # construct a dataframe containing information about the
    # participants
    p_conds = pd\
        .DataFrame\
        .from_dict(conds).T\
        .reset_index()\
        .rename(columns={
            'index': 'pid',
        })
    p_info = pd\
        .DataFrame\
        .from_dict(find_bad_participants(data_path.namebase, data))
    participants = pd.merge(p_conds, p_info, on=['assignment', 'pid'])\
                     .sort('timestamp')\
                     .set_index('timestamp')

    # drop bad participants
    all_pids = p_info.set_index(['assignment', 'pid'])
    bad_pids = all_pids.dropna()
    n_subj = len(all_pids)
    n_good = n_subj - len(bad_pids)
    logger.info(
        "%d/%d (%.1f%%) participants OK",
        n_good, n_subj, n_good * 100. / n_subj)
    data = data\
        .set_index(['assignment', 'pid'])\
        .drop(bad_pids.index)\
        .reset_index()

    # extract the responses and times and make them separate columns,
    # rather than separate phases
    fields = ['psiturk_time', 'response', 'response_time']
    data = data.set_index(
        ['assignment', 'pid', 'mode', 'trial', 'trial_phase'])
    responses = data[fields].unstack('trial_phase')
    data = data.reset_index('trial_phase', drop=True).drop(fields, axis=1)

    data['response'] = responses['response', 'stim']
    data['time'] = responses['response_time', 'stim']
    data['timestamp'] = responses['psiturk_time', 'prestim']
    data['timestamp'] = parse_timestamp(data, 'timestamp')

    # drop duplicated rows and sort the dataframe
    data = data\
        .drop_duplicates()\
        .sortlevel()\
        .reset_index()

    def add_condition(df):
        info = conds[df.name]
        df['condition'] = info['condition']
        # sanity check -- make sure assignment and counterbalance
        # fields match
        assert (df['assignment'] == info['assignment']).all()
        return df

    # add a column for the condition code
    data = data.groupby('pid').apply(add_condition)

    return data, participants


def load_events(data_path):
    """Load experiment event data (e.g. window resizing and the like) from
    the given path. Returns a pandas DataFrame.

    """
    # load the data
    events = pd.read_csv(data_path.joinpath("eventdata_all.csv"))
    # split uniqueid into pid and assignment
    events = split_uniqueid(events, 'uniqueid')
    # parse timestamps
    events['timestamp'] = parse_timestamp(events, 'timestamp')
    # sort by pid/assignment
    events = events\
        .set_index(['assignment', 'pid', 'timestamp'])\
        .reset_index()\
        .sort(['assignment', 'pid', 'timestamp'])

    return events


def save_dpkg(dataset_path, data, meta, events, participants):
    dp = dpkg.DataPackage(name=dataset_path.name, licenses=['odc-by'])
    dp['version'] = '1.0.0'
    dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
    dp.add_contributor("Thomas L. Griffiths", "tom_griffiths@berkeley.edu")

    # add experiment data, and save it as csv
    r1 = dpkg.Resource(
        name="experiment.csv", fmt="csv",
        pth="./experiment.csv", data=data)
    r1['mediaformat'] = 'text/csv'
    dp.add_resource(r1)

    # add metadata, and save it inline as json
    r2 = dpkg.Resource(name="metadata", fmt="json", data=meta)
    r2['mediaformat'] = 'application/json'
    dp.add_resource(r2)

    # add event data, and save it as csv
    r3 = dpkg.Resource(
        name="events.csv", fmt="csv",
        pth="./events.csv", data=events)
    r3['mediaformat'] = 'text/csv'
    dp.add_resource(r3)

    # add participant info, and save it as csv
    r3 = dpkg.Resource(
        name="participants.csv", fmt="csv",
        pth="./participants.csv", data=participants)
    r3['mediaformat'] = 'text/csv'
    dp.add_resource(r3)

    # save the datapackage
    dp.save(dataset_path.dirname())
    logger.info("Saved to '%s'", dataset_path.relpath())


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force processed data to be saved.")

    args = parser.parse_args()

    # paths to the data and where we will save it
    data_path = DATA_PATH.joinpath("human-raw", args.exp)
    dest_path = DATA_PATH.joinpath("human", "%s.dpkg" % args.exp)

    # don't do anything if the datapackage already exists
    if dest_path.exists() and not args.force:
        sys.exit(0)

    # create the directory if it doesn't exist
    if not dest_path.dirname().exists:
        dest_path.dirname().makedirs_p()

    # load the data
    meta, conds, fields = load_meta(data_path)
    data, participants = load_data(data_path, conds, fields)
    events = load_events(data_path)

    # save it
    save_dpkg(dest_path, data, meta, events, participants)
