from path import path
from termcolor import colored
import traceback

from . import __all__
from . import *
from . import util

root = path("../")
config = util.load_config(root.joinpath("config.ini"))
version = config.get("global", "version")
results_path = root.joinpath(config.get("paths", "results"), version)
fig_path = root.joinpath(config.get("paths", "figures"), version)

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    try:
        print func.plot(results_path, fig_path)
    except:
        print colored(traceback.format_exc(limit=3), "red")
