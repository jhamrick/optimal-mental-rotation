import snippets.datapackage as dpkg
import numpy as np
import pandas as pd
import scipy.special
import logging

logger = logging.getLogger("mental_rotation.analysis")


def zscore(x):
    return (x - x.mean()) / x.std()


def modtheta(x):
    if x > 180:
        return 360 - x
    else:
        return x


def load_human(version, data_path):
    human_dp = dpkg.DataPackage.load(
        data_path.joinpath("human", "%s.dpkg" % version))
    exp_all = human_dp.load_resource("experiment.csv")

    # get just the experiment trials
    exp = exp_all\
        .groupby(
            exp_all['mode'].apply(
                lambda x: x.startswith('experiment')))\
        .get_group(True)

    # convert seconds to milliseconds
    exp.loc[:, 'time'] *= 1000

    # remove trials where people took a ridiculous amount of time
    # (i.e., greater than 20 seconds) or they took way too little time
    too_long = exp.index[(exp['time'] > 20000) | (exp['time'] < 100)]
    logger.warn(
        "Excluding %d/%d trials from analysis",
        len(too_long), len(exp))
    exp = exp.drop(too_long)

    # compute correct responses
    exp.loc[:, 'correct'] = exp['flipped'] == exp['response']

    # compute modtheta, where all angles are between 0 and 180
    exp.loc[:, 'modtheta'] = exp['theta'].apply(modtheta)

    # make stimulus names be integers
    exp.loc[:, 'stimulus'] = [int(x) for x in exp['stimulus']]

    # split into the two different blocks
    expA = exp.groupby('mode').get_group('experimentA')
    expB = exp.groupby('mode').get_group('experimentB')
    expB.loc[:, 'trial'] += expA['trial'].max()
    exp.loc[expA.index, 'trial'] = expA['trial']
    exp.loc[expB.index, 'trial'] = expB['trial']

    exp_data = {
        'expA': expA,
        'expB': expB,
        'exp': exp
    }

    return exp_all, exp_data


def load_model(name, version, data_path):
    dp = dpkg.DataPackage.load(
        data_path.joinpath("model", "%s_%s.dpkg" % (name, version)))
    data = dp.load_resource("model.csv")
    data['modtheta'] = data['theta'].apply(modtheta)
    data['correct'] = data['flipped'] == data['hypothesis']
    data['time'] = data['nstep']
    return data


def load_all(version, data_path, human=None):
    data = {
        'gs': load_model("GoldStandardModel", version, data_path),
        'oc': load_model("OracleModel", version, data_path),
        'th': load_model("ThresholdModel", version, data_path),
        'hc': load_model("HillClimbingModel", version, data_path),
    }

    bq = load_model("BayesianQuadratureModel", version, data_path)
    data['bq'] = bq.groupby(['step', 'prior']).get_group((0.6, 0.5))
    data['bqp'] = bq.groupby(['step', 'prior']).get_group((0.6, 0.55))

    data['oc'] = data['oc'].groupby(['step', 'prior']).get_group((0.1, 0.5))
    data['th'] = data['th'].groupby(['step', 'prior']).get_group((0.6, 0.5))
    data['hc'] = data['hc'].groupby(['step', 'prior']).get_group((0.1, 0.5))

    if human is None:
        data.update(load_human(version, data_path)[1])
    else:
        data.update(human)

    return data


def bootstrap_mean(x, nsamples=1000):
    arr = np.asarray(x)
    n, = arr.shape
    boot_idx = np.random.randint(0, n, n * nsamples)
    boot_arr = arr[boot_idx].reshape((n, nsamples))
    boot_mean = boot_arr.mean(axis=0)
    stats = pd.Series(
        np.percentile(boot_mean, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'],
        name=x.name)
    return stats


def bootstrap_median(x, nsamples=1000):
    arr = np.asarray(x)
    n, = arr.shape
    boot_idx = np.random.randint(0, n, n * nsamples)
    boot_arr = arr[boot_idx].reshape((n, nsamples))
    boot_median = np.percentile(boot_arr, 50, axis=0)
    stats = pd.Series(
        np.percentile(boot_median, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'],
        name=x.name)
    return stats


def beta(x, percentiles=None):
    arr = np.asarray(x, dtype=int)
    alpha = float((arr == 1).sum()) + 0.5
    beta = float((arr == 0).sum()) + 0.5
    if percentiles is None:
        lower, mean, upper = scipy.special.btdtri(
            alpha, beta, [0.025, 0.5, 0.975])
        stats = pd.Series(
            [lower, mean, upper],
            index=['lower', 'median', 'upper'],
            name=x.name)
    else:
        stats = pd.Series(
            scipy.special.btdtri(alpha, beta, percentiles),
            index=percentiles,
            name=x.name)
    return stats


def bootcorr(x, y, nsamples=10000, method='pearson'):
    arr1 = np.asarray(x)
    arr2 = np.asarray(y)
    n, = arr1.shape
    assert arr1.shape == arr2.shape

    boot_corr = np.empty(nsamples)

    for i in xrange(nsamples):
        boot_idx = np.random.randint(0, n, n)
        boot_arr1 = arr1[boot_idx]
        boot_arr2 = arr2[boot_idx]

        if method == 'pearson':
            func = scipy.stats.pearsonr
        elif method == 'spearman':
            func = scipy.stats.spearmanr
        else:
            raise ValueError("invalid method: %s" % method)

        ii = ~np.isnan(boot_arr1) & ~np.isnan(boot_arr2)
        boot_corr[i] = func(boot_arr1[ii], boot_arr2[ii])[0]

    stats = pd.Series(
        np.percentile(boot_corr, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'])

    return stats
