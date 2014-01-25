import snippets.datapackage as dpkg
import numpy as np
import pandas as pd
import scipy.special


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

    # remove trials where people took a ridiculous amount of time
    # (i.e., greater than 20 seconds)
    too_long = exp.index[exp['time'] > 20]
    exp = exp.drop(too_long)

    # compute correct responses
    exp['correct'] = exp['flipped'] == exp['response']

    # compute zscored time
    exp['ztime'] = exp.groupby('pid')['time'].apply(zscore)

    # compute modtheta, where all angles are between 0 and 180
    exp['modtheta'] = exp['theta'].apply(modtheta)

    # make stimulus names be integers
    exp['stimulus'] = [int(x) for x in exp['stimulus']]

    # split into the two different blocks
    expA = exp.groupby('mode').get_group('experimentA')
    expB = exp.groupby('mode').get_group('experimentB')
    expB['trial'] += expA['trial'].max()
    exp['trial'].ix[expA.index] = expA['trial']
    exp['trial'].ix[expB.index] = expB['trial']

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
    data['ztime'] = zscore(data['nstep'])
    return data


def load_all(version, data_path):
    data = {
        'gs': load_model("GoldStandardModel", version, data_path),
        'oc': load_model("OracleModel", version, data_path),
        'th': load_model("ThresholdModel", version, data_path),
        'hc': load_model("HillClimbingModel", version, data_path),
        'bq': load_model("BayesianQuadratureModel", version, data_path)
    }
    data.update(load_human(version, data_path)[1])
    return data


def bootstrap(x, nsamples=1000):
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


def beta(x):
    arr = np.asarray(x, dtype=int)
    alpha = float((arr == 1).sum()) + 0.5
    beta = float((arr == 0).sum()) + 0.5
    lower, mean, upper = scipy.special.btdtri(
        alpha, beta, [0.025, 0.5, 0.975])
    stats = pd.Series(
        [lower, mean, upper],
        index=['lower', 'median', 'upper'],
        name=x.name)
    return stats


def bootcorr(x, y, nsamples=1000, method='pearson'):
    arr1 = np.asarray(x)
    arr2 = np.asarray(y)
    n, = arr1.shape
    assert arr1.shape == arr2.shape

    boot_idx = np.random.randint(0, n, n * nsamples)
    boot_arr1 = arr1[boot_idx].reshape((n, nsamples))
    boot_arr2 = arr2[boot_idx].reshape((n, nsamples))
    boot_corr = np.empty(nsamples)

    for i in xrange(nsamples):
        ii = ~np.isnan(boot_arr1[:, i]) & ~np.isnan(boot_arr2[:, i])
        if method == 'pearson':
            func = scipy.stats.pearsonr
        elif method == 'spearman':
            func = scipy.stats.spearmanr
        else:
            raise ValueError("invalid method: %s" % method)

        boot_corr[i] = func(
            boot_arr1[ii, i],
            boot_arr2[ii, i])[0]

    stats = pd.Series(
        np.percentile(boot_corr, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'])

    return stats
