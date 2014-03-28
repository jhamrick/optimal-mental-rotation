from path import path
from termcolor import colored
import pandas as pd

from . import __all__
from . import *
from . import util

root = path("../")
config = util.load_config(root.joinpath("config.ini"))
version = config.get("global", "version")
data_path = root.joinpath(config.get("paths", "data"))
data = util.load_all(version, data_path)
results_path = root.joinpath(config.get("paths", "results"), version)
seed = config.getint("global", "seed")

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    pth = func.run(data, results_path, seed)
    print pth
    if pth.ext == '.csv':
        df = pd.read_csv(pth, index_col='model')
        print df
