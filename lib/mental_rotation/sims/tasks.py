from path import path
from itertools import product
import json
import numpy as np


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

    @staticmethod
    def _make_chunks(model_opts, num_samples, chunksize):
        # make the list of parameter names that we want for each simulation
        opt_names = sorted(model_opts.keys())
        param_names = opt_names + ['sample']

        # make a list of lists, where each inner list contains the
        # possible parameter values for one particular parameter
        param_vals = [model_opts[key] for key in param_names[:-1]]
        param_vals.append(range(num_samples))

        # make a list of all parameter combinations
        combs = np.array(list(product(*param_vals)))

        # compute the number of chunks
        if chunksize == -1:
            n_chunks = 1
        else:
            n_chunks = int(np.ceil(combs.shape[0] / chunksize))

        ichunks = np.array_split(
            np.arange(len(combs)), n_chunks, axis=0)

        chunks = []
        for idx, ichunk in enumerate(ichunks):
            chunk = {sim: dict(zip(param_names, combs[sim])) for sim in ichunk}
            chunks.append(chunk)

        return chunks

    @classmethod
    def create(cls, params):
        """Create the tasks dictionary from the parameters."""

        sim_root = path(params["sim_root"])
        if not sim_root.exists():
            sim_root.makedirs_p()

        stim_paths = params['stim_paths']
        num_samples = params['num_samples']
        chunksize = float(params['chunksize'])
        opts = params['model_opts']

        tasks = cls()
        completed = cls()
        for istim, stim in enumerate(stim_paths):
            stim = path(stim)
            rot = float(stim.namebase.split("_")[1])
            if rot in (0, 180):
                chunks = cls._make_chunks(opts, num_samples * 2, chunksize)
            else:
                chunks = cls._make_chunks(opts, num_samples, chunksize)

            for idx, chunk in enumerate(chunks):
                task_name = "%s~%d" % (stim.namebase, idx)
                data_path = sim_root.joinpath(task_name)

                # Make the task dicts for this sample.
                tasks[task_name] = {
                    "model": params["model"],
                    "istim": istim,
                    "stim_path": str(stim),
                    "data_path": str(data_path),
                    "task_name": task_name,
                    "seed": abs(hash(task_name)),
                    "model_opts": chunk,
                }

                completed[task_name] = False

        return tasks, completed
