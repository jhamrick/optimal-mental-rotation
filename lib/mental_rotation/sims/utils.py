import json
import numpy as np
from mental_rotation import SIM_PATH
from mental_rotation import SIM_SCRIPT_PATH as SCRIPT_PATH


def parse_address(address):
    host, port = address.split(":")
    return host, int(port)


def get_params(model, exp):
    """Load the parameters from the simulation script."""
    sim_root = SIM_PATH.joinpath(model, exp)
    script_root = SCRIPT_PATH.joinpath(model)
    script_file = script_root.joinpath("%s.json" % exp)
    with script_file.open("r") as fid:
        script = json.load(fid)
    script["script_root"] = str(script_root)
    script["sim_root"] = str(sim_root)
    script["tasks_path"] = str(sim_root.joinpath("tasks.json"))
    return script
