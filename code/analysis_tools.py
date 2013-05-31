import numpy as np
import scipy.stats
import pandas as pd
import util
import circstats as circ

from snippets.stats import xcorr


def new_cmd(name, value):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, value)
    return cmd + "\n"


def linreg(A, B):
    A_ = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    x, res, rank, s = np.linalg.lstsq(A_, B)
    return x


def load_stims():
    stims = util.find_stims()
    true_hyp = np.array([int(s[-1]) for s in stims])
    thetas = np.array([util.load_stimulus(s)[0] for s in stims])
    min_ang = np.abs(circ.wrapdiff(thetas))
    unique_ang = np.unique(np.round(min_ang, decimals=8))
    ih0 = ~(true_hyp.astype('bool'))
    ih1 = true_hyp.astype('bool')

    sdata = {
        'stims': stims,
        'true_hyp': true_hyp,
        'thetas': thetas,
        'min_ang': min_ang,
        'unique_ang': unique_ang,
        'ih0': ih0,
        'ih1': ih1,
    }
    return sdata


def load_data(models, stims):
    data = {}
    for model in models:
        data[model] = util.load_sims(model)
        assert (np.array(stims) == np.array(data[model][0])).all()
    return data


def process_Z(data):
    Z = data[2]
    if Z.shape[1] > 1:
        raise AssertionError("more than one sample")
    Z_mean = Z[:, 0, 0]
    Z_std = np.sqrt(Z[:, 0, 1:])
    return Z_mean, Z_std


def calc_ME(models, data, sdata):
    true_hyp = sdata['true_hyp']
    ih0 = sdata['ih0']
    ih1 = sdata['ih1']
    ME = {}
    for model in models:
        hyp = data[model][4]
        err = np.abs(hyp - true_hyp[:, None])
        ME_all = np.mean(err)
        ME_h0 = np.mean(err[ih0])
        ME_h1 = np.mean(err[ih1])
        ME[model] = ME_all, ME_h0, ME_h1
    return ME


def ME_latex(models, ME, shortnames):
    latex = []
    for model in models:
        shortname = shortnames[model]
        ME_all, ME_h0, ME_h1 = ME[model]
        latex.extend([
            new_cmd("%sME" % shortname, r"\ME{}=%.2f" % ME_all),
            new_cmd("%sMEdiff" % shortname, r"\ME{}=%.2f" % ME_h0),
            new_cmd("%sMEsame" % shortname, r"\ME{}=%.2f" % ME_h1)
        ])
    return "".join(latex)


def ME_table(models, ME, shortnames):
    tbl = [ME[model] for model in models]
    index = [shortnames[model] for model in models]
    df = pd.DataFrame(
        tbl,
        index=index,
        columns=("all", "h0", "h1"))
    df.columns.name = "ME"
    return df


def calc_Rot(models, data, sdata, only_correct=False):

    Rot_diff = {}
    Rot_same = {}

    true_hyp = sdata['true_hyp']
    unique_ang = sdata['unique_ang']
    min_ang = sdata['min_ang']
    ih0 = sdata['ih0']
    ih1 = sdata['ih1']

    for model in models:
        samps = data[model][1]
        hyp = data[model][4]
        err = np.abs(hyp - true_hyp[:, None])

        if only_correct:
            correct = err == 0
        else:
            correct = np.ones(err.shape).astype('bool')

        # calculate for different pairs
        dd = samps[ih0[:, None] & correct] * 100
        diff_raw = samps[ih0[:, None]] * 100
        diff_mean = np.mean(dd)
        diff_std = np.std(dd, ddof=1)
        Rot_diff[model] = (diff_raw, diff_mean, diff_std)

        # calculate for same pairs
        same_raw = []
        same_mean = np.empty(unique_ang.size)
        same_std = np.empty(unique_ang.size)
        for aidx, ang in enumerate(unique_ang):
            idx = (np.abs(min_ang - ang) < 1e-8).ravel()
            ss = samps[(idx & ih1)[:, None] & correct] * 100
            same_raw.append(samps[(idx & ih1)[:, None]] * 100)
            same_mean[aidx] = np.mean(ss)
            same_std[aidx] = np.std(ss, ddof=1)
        Rot_same[model] = (same_raw, same_mean, same_std)

    return Rot_diff, Rot_same


def calc_Corr(models, sdata, Rot_same):
    unique_ang = sdata['unique_ang']
    sm = unique_ang < (np.pi / 2.)
    Corr = {}
    for model in models:
        raw, mean, std = Rot_same[model]
        corr = float(xcorr(unique_ang, mean))
        corr_sm = xcorr(unique_ang[sm], mean[sm])
        corr_bg = xcorr(unique_ang[~sm], mean[~sm])
        Corr[model] = (corr, corr_sm, corr_bg)
    return Corr


def Corr_latex(models, Corr, shortnames):
    latex = []
    for model in models:
        shortname = shortnames[model]
        corr_all, corr_sm, corr_bg = Corr[model]
        latex.extend([
            new_cmd("%scorr" % shortname, r"\rho=%.2f" % corr_all),
            new_cmd("%scorrsm" % shortname, r"\rho=%.2f" % corr_sm),
            new_cmd("%scorrbg" % shortname, r"\rho=%.2f" % corr_bg)
        ])
    return "".join(latex)


def Corr_table(models, Corr, shortnames):
    tbl = [Corr[model] for model in models]
    index = [shortnames[model] for model in models]
    df = pd.DataFrame(
        tbl,
        index=index,
        columns=("all", "< 90", ">= 90"))
    df.columns.name = "Correlation"
    return df


def calc_MSE(models, data):
    MSE = {}
    gs_Z_mean, gs_Z_std = process_Z(data["GoldStandardModel"])
    for model in models:
        Z_mean, Z_std = process_Z(data[model])
        normed_err = (gs_Z_mean - Z_mean) / np.ptp(gs_Z_mean)
        MSE[model] = np.mean(normed_err ** 2)
    return MSE


def MSE_latex(models, MSE, shortnames):
    latex = []
    for model in models:
        shortname = shortnames[model]
        mse = MSE[model]
        latex.append(new_cmd("%sMSE" % shortname, r"\MSE{}=%.2f" % mse))
    return "".join(latex)


def MSE_table(models, MSE, shortnames):
    tbl = [[MSE[model]] for model in models]
    index = [shortnames[model] for model in models]
    df = pd.DataFrame(
        tbl,
        index=index,
        columns=("MSE",))
    return df


def format_latex(latex):
    texlist = []
    texlist.append(latex['head'])
    for section in sorted(latex):
        if section == 'head':
            continue
        texlist.append("%%%% %s" % section)
        texlist.append(latex[section])
    texlist.append("")
    return "\n".join(texlist)
