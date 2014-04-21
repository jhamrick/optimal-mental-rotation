import json
import logging
import multiprocessing as mp
import numpy as np
import signal
import sys
import tempfile
import traceback
import time


from datetime import datetime
from path import path
from xmlrpclib import ServerProxy

import mental_rotation.model as m
from mental_rotation.stimulus import Stimulus2D
from .util import run_command

logger = logging.getLogger("mental_rotation.sims.client")


def signal_handler(signal, frame):
    mp.util.debug("Keyboard interrupt!")
    sys.exit(-2)

signal.signal(signal.SIGINT, signal_handler)


def simulate(task):
    stim_path = task["stim_path"]
    model_name = task["model"]
    seed = task["seed"]
    data_path = path(task["data_path"])
    model_opts = task["model_opts"]

    X = Stimulus2D.load(stim_path)
    Xa = X.copy_from_initial().vertices
    Xb = X.copy_from_vertices().vertices

    try:
        model_class = getattr(m, model_name)
    except AttributeError:
        raise ValueError("unhandled model: %s" % model_name)

    np.random.seed(seed)

    for iopt, opts in model_opts.iteritems():
        logger.info("Task '%s', part %s", task['task_name'], iopt)
        model = model_class(Xa, Xb, **opts)
        try:
            model.sample()
        except SystemExit:
            raise
        except:
            print traceback.format_exc()

        model.save(data_path, "part_%s" % iopt, force=True)


def worker_job(host, port):
    # connet to the server
    pandaserver = ServerProxy("http://%s:%d" % (host, port))
    while True:
        try:
            sim_root = path(pandaserver.panda_connect())
        except:
            time.sleep(1)
        else:
            break

    tmpdir = path(tempfile.mkdtemp())

    while True:
        # get the next task from the pandaserver
        while True:
            try:
                task = pandaserver.panda_request()
            except:
                time.sleep(1)
            else:
                break

        # no more tasks left
        if task is None:
            logger.info("Nothing left to do, shutting down")
            break

        task = json.loads(task)
        task_name = task["task_name"]
        task["data_path"] = tmpdir.joinpath("%s.h5" % task_name)

        logger.info("Got task '%s'", task_name)
        startime = datetime.now()
        error = simulate(task)
        dt = (datetime.now() - startime).total_seconds()
        logger.info("Completed task '%s' in %s seconds", task_name, dt)

        if error is not None:
            logger.error("Task '%s' failed with error:\n%s", task_name, error)
            while True:
                try:
                    pandaserver.panda_error(task_name)
                except:
                    time.sleep(1)
                else:
                    break

        else:
            data_path = path(task["data_path"])

            # then send it to the server
            if host in ('localhost', '127.0.0.1'):
                prefix = ""
                dst_path = "%s/%s.h5" % (sim_root, task_name)
                options = []
            else:
                prefix = "ubuntu@%s:" % host
                dst_path = sim_root.joinpath("%s.h5" % task_name)
                options = [
                    '-o', 'StrictHostKeyChecking=no',
                    '-o', 'UserKnownHostsFile=/dev/null'
                ]

            # build the scp command
            cmd = ['scp', '-q']
            cmd.extend(options)
            cmd.append(data_path)
            cmd.append(prefix + dst_path)

            while True:
                try:
                    run_command(logger, cmd)
                except:
                    time.sleep(1)
                else:
                    break

            # then mark it as complete
            while True:
                try:
                    pandaserver.panda_complete(task_name)
                except:
                    time.sleep(1)
                else:
                    break

        task['data_path'].rmtree_p()

    tmpdir.rmtree_p()


def run(host, port, nprocess, loglevel):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(loglevel)
    logger.setLevel(loglevel)
    np.seterr(invalid='ignore')

    # create the worker processes
    if nprocess == 1:
        worker_job(host, port)

    else:
        processes = []
        for i in xrange(nprocess):
            p = mp.Process(target=worker_job, args=(host, port))
            processes.append(p)
            p.start()

        # wait for them to finish
        for p in processes:
            p.join()

    logger.info("Worker threads done, shutting down.")
