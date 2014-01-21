#!/usr/bin/env python

import multiprocessing as mp
import logging
import numpy as np
import signal
import sys
import itertools

from path import path
from datetime import datetime, timedelta

import mental_rotation.model as m
from mental_rotation.stimulus import Stimulus2D
from mental_rotation import MODELS
from tasks import Tasks


logger = logging.getLogger("mental_rotation.sims")


def signal_handler(signal, frame):
    mp.util.debug("Keyboard interrupt!")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)


def load_tasks(params, force):
    tasks_file = path(params["tasks_path"])
    completed_file = path(params["completed_path"])

    if tasks_file.exists() and completed_file.exists() and not force:
        tasks = Tasks.load(tasks_file)
        completed = Tasks.load(completed_file)
    else:
        tasks, completed = Tasks.create(params)
        tasks.save(tasks_file)
        completed.save(completed_file)

    logger.info("%d tasks loaded", len(tasks))
    return tasks, completed


def simulate(task):
    samples = task["samples"]
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
    if not data_path.exists():
        data_path.makedirs()

    for isamp in samples:
        dest = data_path.joinpath("sample_%02d" % isamp)
        model = model_class(Xa, Xb, **model_opts)
        model.sample()

        if dest.exists():
            dest.rmtree_p()
        model.save(dest)

    return task["task_name"]


def report(task_name, num_finished, num_tasks, start_time):
    progress = 100 * float(num_finished) / num_tasks
    dt = datetime.now() - start_time
    avg_dt = timedelta(
        seconds=(dt.total_seconds() / float(num_finished + 1e-5)))
    time_left = timedelta(
        seconds=(avg_dt.total_seconds() * (num_tasks - num_finished)))

    logger.info("-" * 40)
    logger.info("Task `%s` complete", task_name)
    logger.info("Progress: %d/%d (%.2f%%)",
                num_finished, num_tasks, progress)
    logger.info("Time elapsed  : %s", str(dt))
    logger.info("Time per task : %s", str(avg_dt))
    logger.info("Time remaining: %s", str(time_left))


def queue_tasks(tasks, completed, force):
    queued_tasks = []
    for task_name in sorted(tasks.keys()):
        task = tasks[task_name]
        complete = completed[task_name]
        if force or not complete:
            queued_tasks.append(task)

    logger.info("%d tasks queued", len(queued_tasks))
    return queued_tasks


def run(params, force):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(params['loglevel'])
    logger.setLevel(params['loglevel'])

    # record the starting time
    start_time = datetime.now()

    # load tasks and put eligible ones in the queue
    tasks, completed = load_tasks(params, force)
    queued_tasks = queue_tasks(tasks, completed, force)
    num_tasks = len(queued_tasks)
    completed_file = path(params["completed_path"])

    # create the pool of worker processes
    pool = mp.Pool()
    results = pool.imap_unordered(simulate, queued_tasks)

    # process tasks as they are completed
    for i, task_name in enumerate(results):
        # mark it done
        completed[task_name] = True
        completed.save(completed_file)

        # report progress
        report(task_name, i, num_tasks, start_time)

    # done
    logger.info("Jobs complete. Shutting down.")
