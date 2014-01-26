from path import path
from termcolor import colored

from . import __all__
from . import *
from . import util

root = path("../")
config = util.load_config(root.joinpath("config.ini"))
version = config.get("global", "version")
data_path = root.joinpath(config.get("paths", "data"))
data = util.load_all(version, data_path)
fig_path = root.joinpath(config.get("paths", "figures"), version)
seed = config.getint("global", "seed")

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    print func.plot(data, fig_path, seed)
