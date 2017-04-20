from ConfigParser import SafeConfigParser
import path
import pandas as pd

from mental_rotation.analysis import load_human, load_model, load_all
from mental_rotation.analysis import beta, bootcorr, modtheta
from mental_rotation.analysis import bootstrap_median, bootstrap_mean


def load_config(pth):
    config = SafeConfigParser()
    config.read(pth)
    return config


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"


report_spearman = "\rho={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_spearman = r"$\rho={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_pearson = "r={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_pearson = r"$r={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_percent = "M={median:.1f}%, 95% CI [{lower:.1f}%, {upper:.1f}%]"
latex_percent = r"$M={median:.1f}\%$, 95\% CI $[{lower:.1f}\%, {upper:.1f}\%]$"

report_mean = "M={median:.1f}, 95% CI [{lower:.1f}, {upper:.1f}]"
latex_mean = r"$M={median:.1f}$, 95\% CI $[{lower:.1f}, {upper:.1f}]$"

report_msec = "M={median:.1f} msec, 95% CI [{lower:.1f} msec, {upper:.1f} msec]"
latex_msec = r"$M={median:.1f}$ msec, 95\% CI $[{lower:.1f}\textrm{{ msec}}, {upper:.1f}\textrm{{ msec}}]$"


def run_analysis(func):
    config = load_config("config.ini")
    version = config.get("global", "version")
    data_path = path.Path(config.get("paths", "data"))
    data = load_all(version, data_path)
    results_path = path.Path(config.get("paths", "results")).joinpath(version)
    seed = config.getint("global", "seed")
    pth = func(data, results_path, seed)
    print pth
    if pth.ext == ".csv":
        df = pd.read_csv(pth, index_col='model')
        print df
