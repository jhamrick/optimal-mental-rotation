import numpy as np
import pandas as pd
import util


def new_cmd(name, value):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, value)
    return cmd + "\n"


def linreg(A, B):
    A_ = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    x, res, rank, s = np.linalg.lstsq(A_, B)
    return x


def hypidx(true_hyp):
    ih0 = ~(true_hyp.astype('bool'))
    ih1 = true_hyp.astype('bool')
    return ih0, ih1


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
    if Z.shape[1] > 1:
        Z_std = np.sqrt(Z[:, 0, 1:])
    else:
        Z_std = None
    return Z_mean, Z_std


def calc_ME(models, true_hyp, data):
    ME = {}
    for model in models:
        ih0, ih1 = hypidx(true_hyp)
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
    tbl = []
    index = []
    for model in models:
        tbl.append(ME[model])
        index.append(shortnames[model])

    df = pd.DataFrame(
        tbl,
        index=index,
        columns=("all", "h0", "h1"))
    return df
