import multiprocessing as mp
import logging
import signal
import sys
import json
import socket

from path import path
from datetime import datetime, timedelta

from .tasks import Tasks
from .util import run_command


logger = logging.getLogger("mental_rotation.sims.manager")


def signal_handler(signal, frame):
    mp.util.debug("Keyboard interrupt!")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)


def get_next_task(tasks, queued):
    if len(queued) == 0:
        return None
    task_name = queued.pop(0)
    task = tasks[task_name]
    queued.append(task_name)
    return task


def extract_result(task):
    task_name = task['task_name']
    data_path = path(task['data_path'])
    loc = data_path.dirname().joinpath("%s.tar.gz" % task_name)
    if not loc.exists():
        logger.error("data not found for task '%s'")
        return False

    if data_path.exists():
        data_path.rmtree_p()

    cmd = ['tar', '-xf', loc, '-C', data_path.dirname()]
    run_command(logger, cmd)

    loc.remove()
    return True


def complete_task(params, task_name, completed, queued):
    completed_file = path(params["completed_path"])

    # mark it done
    completed[task_name] = True
    completed.save(completed_file)

    # remove it from open tasks
    if task_name in queued:
        queued.remove(task_name)


def create_tasks(params, force):
    tasks_file = path(params["tasks_path"])
    completed_file = path(params["completed_path"])

    if tasks_file.exists() and completed_file.exists() and not force:
        logger.info("tasks file already exists")

    else:
        tasks, completed = Tasks.create(params)
        tasks.save(tasks_file)
        completed.save(completed_file)

        logger.info("tasks file created")


def load_tasks(params, force):
    tasks_file = path(params["tasks_path"])
    completed_file = path(params["completed_path"])

    tasks = Tasks.load(tasks_file)
    completed = Tasks.load(completed_file)

    logger.info("%d tasks loaded", len(tasks))

    queued = []
    for task_name in sorted(tasks.keys()):
        complete = completed[task_name]
        if force or not complete:
            queued.append(task_name)

    logger.info("%d tasks queued", len(queued))
    return tasks, completed, queued


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


def run(params, force):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(params['loglevel'])
    logger.setLevel(params['loglevel'])

    # record the starting time
    start_time = datetime.now()

    # load tasks and put eligible ones in the queue
    create_tasks(params, force)
    tasks, completed, queued = load_tasks(params, force)
    errors = []
    num_tasks = len(queued)

    port = 55556
    backlog = 1024
    size = 1024

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen(backlog)

    i = 0
    while True:
        client, address = s.accept()
        data = client.recv(size)

        print "connected to client"

        # send the client the path to save the data
        if data.rstrip() == 'panda_connect':
            client.send(str(path(params['sim_root']).abspath()))

        # process a request for a new task
        elif data.rstrip() == 'panda_request':
            task = get_next_task(tasks, queued)

            if task:
                # send the task
                client.send(json.dumps(task))
            else:
                # no more tasks left
                client.send('no_panda')

        # a client is done with a task, so mark it complete
        elif data.startswith('panda_complete'):
            task_name = data.replace('panda_complete', '').rstrip()
            if extract_result(tasks[task_name]):
                complete_task(params, task_name, completed, queued)

                # report progress
                report(task_name, i, num_tasks, start_time)
                i += 1

        # there was an error
        elif data.startswith('panda_error'):
            task_name = data.replace('panda_error', '').rstrip()

            logger.error("Task '%s' failed with an error", task_name)
            errors.append(task_name)

        # reload the tasks list
        elif data.rstrip() == 'panda_reload':
            tasks, completed, queued = load_tasks(params, False)

        else:
            logger.error("Unrecognized command: %s", data)

        client.close()

    if len(errors) > 0:
        # report any errors
        logger.error("The following tasks had errors: %s", errors)
        logger.info("Shutting down.")

    else:
        # done
        logger.info("Jobs complete. Shutting down.")
