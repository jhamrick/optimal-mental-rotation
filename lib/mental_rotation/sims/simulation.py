from datetime import datetime, timedelta
from multiprocessing import Process
from path import path
import multiprocessing as mp
import numpy as np
import sys

import mental_rotation
from mental_rotation.stimulus import Stimulus2D
import mental_rotation.model as m


class Simulation(Process):
    """Simulation job."""

    def __init__(self, task, params, info_lock, save=False):
        self.task = task
        self.params = params
        self.info_lock = info_lock
        self.save = save
        self.start_time = None
        self.end_time = None
        super(Simulation, self).__init__()

    def simulate(self):

        ## Assorted parameters.
        istim = self.task["istim"]

        stim_path = self.task["stim_path"]
        X = Stimulus2D.load(stim_path)
        Xa = X.copy_from_initial().vertices
        Xb = X.copy_from_vertices().vertices

        model_name = self.task["model"]
        try:
            model_class = getattr(m, model_name)
        except AttributeError:
            raise ValueError("unhandled model: %s" % model_name)

        np.random.seed(self.task["seed"])
        model = model_class(Xa, Xb)
        model.sample()

        if self.save:
            # Write data to file.
            data_path = path(self.task["data_path"])
            if not data_path.dirname().exists():
                data_path.dirname().makedirs()
            model.save(data_path)

            # Mark simulation as complete.
            self.task["complete"] = True

    def print_info(self):
        self.info_lock.acquire()
        dt = self.end_time - self.start_time

        mp.util.info("Total time : %s" % str(dt))

        sys.stdout.flush()
        self.info_lock.release()

    def run(self):
        """Run one simulation."""

        try:
            self.start_time = datetime.now()
            self.simulate()

        except KeyboardInterrupt:
            mp.util.debug("Keyboard interrupt!")
            sys.exit(100)

        except Exception as err:
            mp.util.debug("Error: %s" % err)
            raise

        else:
            self.end_time = datetime.now()
            # print out information about the task in a thread-safe
            # manner (so information isn't interleaved across tasks)
            self.print_info()
