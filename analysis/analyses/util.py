from ConfigParser import SafeConfigParser

from mental_rotation.analysis import load_human, load_model, load_all
from mental_rotation.analysis import bootstrap, beta


def load_config(pth):
    config = SafeConfigParser()
    config.read(pth)
    return config


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"
