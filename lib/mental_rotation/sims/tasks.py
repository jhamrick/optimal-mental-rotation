from itertools import product
import json
import numpy as np
import path


class Tasks(dict):

    def save(self, filename):
        with open(path.Path(filename), "w") as fh:
            json.dump(self, fh)

    @classmethod
    def load(cls, filename):
        tasks = cls()
        with open(path.Path(filename), "r") as fh:
            tasks.update(json.load(fh))
        return tasks

    @classmethod
    def create(cls, params):
        """Create the tasks dictionary from the parameters."""

        sim_root = path.Path(params["sim_root"])
        if not sim_root.exists():
            sim_root.makedirs_p()

        stim_paths = params['stim_paths']
        num_samples = params['num_samples']
        chunksize = float(params['chunksize'])
        opts = params['model_opts']

        tasks = cls()
        completed = cls()
        for istim, stim in enumerate(stim_paths):
            stim = path.Path(stim)
            rot = float(stim.namebase.split("_")[1])
            if rot in (0, 180):
                ns = num_samples * 2
            else:
                ns = num_samples

            opt_names = sorted(opts.keys())
            param_names = opt_names + ['sample']
            param_vals = [opts[key] for key in param_names[:-1]]
            param_vals.append(range(ns))

            combs = np.array(list(product(*param_vals)))
            if chunksize == -1:
                n_chunks = 1
            else:
                n_chunks = int(np.ceil(combs.shape[0] / chunksize))

            ichunks = np.array_split(
                np.arange(len(combs)), n_chunks, axis=0)

            for idx, ichunk in enumerate(ichunks):
                sim_name = "%s~%d" % (stim.namebase, idx)
                data_path = sim_root.joinpath(sim_name)
                model_opts = {
                    i: dict(zip(param_names, combs[i])) for i in ichunk}

                # Make the task dicts for this sample.
                tasks[sim_name] = {
                    "model": params["model"],
                    "istim": istim,
                    "stim_path": str(stim),
                    "data_path": str(data_path),
                    "task_name": sim_name,
                    "seed": abs(hash(sim_name)),
                    "model_opts": model_opts,
                }

                completed[sim_name] = False

        return tasks, completed
