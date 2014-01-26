from ConfigParser import SafeConfigParser

from mental_rotation.analysis import load_human, load_model, load_all
from mental_rotation.analysis import bootstrap, beta, bootcorr


def load_config(pth):
    config = SafeConfigParser()
    config.read(pth)
    return config


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"


report_spearman = "rs={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_spearman = r"$r_s={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_pearson = "r={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_pearson = r"$r={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_percent = "M={median:.1f}%, 95% CI [{lower:.1f}%, {upper:.1f}%]"
latex_percent = r"$M={median:.1f}\%$, 95\% CI $[{lower:.1f}\%, {upper:.1f}\%]$"

report_mean = "M={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_mean = r"$M={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"
