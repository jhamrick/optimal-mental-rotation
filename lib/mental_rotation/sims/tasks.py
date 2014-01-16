from itertools import product as iproduct
from path import path
import json
import numpy as np
from mental_rotation import STIM_PATH


class Tasks(dict):

    def save(self, filename):
        with open(path(filename), "w") as fh:
            json.dump(self, fh)

    @classmethod
    def load(cls, filename):
        tasks = cls()
        with open(path(filename), "r") as fh:
            tasks.update(json.load(fh))
        return tasks

    @classmethod
    def create(cls, params):
        """Create the tasks dictionary from the parameters."""

        sim_root = path(params["sim_root"])
        if not sim_root.exists():
            sim_root.makedirs_p()

        stim_paths = [STIM_PATH.joinpath(x) for x in params['stim_paths']]
        num_samples = params['num_samples']
        chunksize = float(params['chunksize'])

        tasks = cls()
        completed = cls()
        for istim, stim in enumerate(stim_paths):
            rot = float(stim.namebase.split("_")[1])
            if rot in (0, 180):
                ns = num_samples * 2
            else:
                ns = num_samples

            n_chunks = int(np.ceil(ns / chunksize))
            chunks = np.array_split(np.arange(ns), n_chunks, axis=0)

            for ichunk, chunk in enumerate(chunks):
                sim_name = "%s_%02d" % (stim.namebase, ichunk)
                data_path = sim_root.joinpath(stim.namebase)

                # Make the task dicts for this sample.
                tasks[sim_name] = {
                    "model": params["model"],
                    "istim": istim,
                    "stim_path": str(stim),
                    "data_path": str(data_path),
                    "script_root": params["script_root"],
                    "task_name": sim_name,
                    "seed": abs(hash(sim_name)),
                    "num_tries": 0,
                    "samples": [int(x) for x in chunk],
                }

                completed[sim_name] = False

        return tasks, completed
