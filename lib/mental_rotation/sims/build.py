"""Generate IPE simulation scripts."""

# Built-in
import json
import logging
# External
import numpy as np
# Local
from mental_rotation import STIM_PATH, SIM_PATH
from mental_rotation import SIM_SCRIPT_PATH as SCRIPT_PATH

logger = logging.getLogger("mental_rotation.sims")


def build(model, exp, **params):
    """Create a simulation script."""

    force = params['force']

    # Path where we will save the simulations
    sim_root = SIM_PATH.joinpath(model, exp)

    # Path where we will save the simulation script/resources
    script_root = SCRIPT_PATH.joinpath(model)
    script_file = script_root.joinpath("%s.json" % exp)

    # check to see if we would override existing data
    if not force and script_file.exists():
        logger.debug("Script %s already exists", script_file.relpath())
        return

    # remove existing files, if we're overwriting
    if script_root.exists():
        script_root.rmtree()

    # Locations of stimuli
    stim_paths = STIM_PATH.joinpath(params['stim_path']).listdir()

    # Put it all together in a big dictionary...
    script = {}

    # Simulation version
    script['model'] = model
    script['exp'] = exp

    # Various paths -- but strip away the absolute parts, because we
    # might be running the simulations on another computer
    script['sim_root'] = str(sim_root.relpath(SIM_PATH))
    script['stim_paths'] = [str(x.relpath(STIM_PATH)) for x in stim_paths]

    # create the directory for our script and resources, and save them
    if not script_root.exists():
        script_root.makedirs_p()

    with script_file.open("w") as fh:
        json.dump(script, fh, indent=2)

    logger.info("Saved script to %s", script_file.relpath())

    return script
