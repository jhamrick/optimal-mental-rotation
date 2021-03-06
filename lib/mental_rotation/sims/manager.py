import logging
import json
import multiprocessing as mp
import Queue
import path

from datetime import datetime, timedelta
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SocketServer import ThreadingMixIn

from .tasks import Tasks
from .util import run_command


logger = logging.getLogger("mental_rotation.sims.manager")


class TaskManager(object):

    def __init__(self, params, force):
        self.params = params
        self.force = force

        self.tasks = None
        self.completed = None
        self.queue = mp.Queue()
        self.info_lock = mp.Lock()
        self.save_lock = mp.Lock()

        self.create_tasks()
        self.load_tasks()
        self.total_finished = 0

        self.start_time = datetime.now()
        self.errors = []

    def create_tasks(self):
        logger.debug("Creating tasks...")

        tasks_file = path.Path(self.params["tasks_path"])
        completed_file = path.Path(self.params["completed_path"])

        if tasks_file.exists() and completed_file.exists() and not self.force:
            logger.debug("Note: tasks file already exists")

        else:
            tasks, completed = Tasks.create(self.params)

            self.save_lock.acquire()
            tasks.save(tasks_file)
            completed.save(completed_file)
            self.save_lock.release()

            logger.debug("Tasks file created")

    def load_tasks(self):
        logger.debug("Loading tasks...")

        tasks_file = path.Path(self.params["tasks_path"])
        completed_file = path.Path(self.params["completed_path"])

        self.save_lock.acquire()
        self.tasks = Tasks.load(tasks_file)
        self.completed = Tasks.load(completed_file)
        self.save_lock.release()

        logger.info("%d tasks loaded", len(self.tasks))

        self.num_tasks = 0
        self.num_finished = 0
        for task_name in sorted(self.tasks.keys()):
            complete = self.completed[task_name]
            if not complete:
                self.queue.put(task_name)
                self.num_tasks += 1

        logger.info("%d tasks queued", self.num_tasks)

    def get_sim_root(self):
        logger.debug("Got request for sim_root")

        # send the client the path to save the data
        return str(path.Path(self.params['sim_root']).abspath())

    def get_next_task(self):
        # process a request for a new task
        try:
            task_name = self.queue.get(True, timeout=1)
        except Queue.Empty:
            return None

        task = self.tasks[task_name]
        if self.completed[task_name]:
            raise RuntimeError(
                "task '%s' has already been completed!" % task_name)

        logger.debug("Allocating task '%s'", task_name)
        return json.dumps(task)

    def set_complete(self, task_name):
        if self.completed[task_name]:
            raise RuntimeError(
                "task '%s' has already been completed!" % task_name)

        # mark the task as done
        completed_file = path.Path(self.params["completed_path"])
        self.completed[task_name] = True
        self.save_lock.acquire()
        self.completed.save(completed_file)
        self.save_lock.release()

        # report the progress
        self.num_finished += 1
        self.total_finished += 1
        self.report(task_name)

    def set_error(self, task_name):
        logger.error("Task '%s' failed with an error", task_name)
        self.errors.append(task_name)

    def get_status(self):
        progress = 100 * float(self.num_finished) / self.num_tasks
        dt = datetime.now() - self.start_time
        avg_dt = timedelta(
            seconds=(dt.total_seconds() / float(self.total_finished + 1e-5)))
        time_left = timedelta(
            seconds=(avg_dt.total_seconds() * (
                self.num_tasks - self.num_finished)))

        msg = "\n".join([
            "Progress: %d/%d (%.2f%%)" % (
                self.num_finished, self.num_tasks, progress),
            "Time elapsed  : %s" % str(dt),
            "Time per task : %s" % str(avg_dt),
            "Time remaining: %s" % str(time_left)
        ])

        return msg

    def report(self, task_name):
        self.info_lock.acquire()
        logger.info("=" * 40)
        logger.info("Task `%s` complete", task_name)
        for line in self.get_status().split("\n"):
            logger.info(line)
        logger.info("-" * 40)
        self.info_lock.release()


class TaskManagerServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def run(host, port, params, force):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(params['loglevel'])
    logger.setLevel(params['loglevel'])

    if logger.getEffectiveLevel() <= 10:
        logRequests = True
    else:
        logRequests = False

    # create the server
    manager = TaskManager(params, force)
    server = TaskManagerServer(
        (host, port), logRequests=logRequests, allow_none=True)

    server.register_function(manager.load_tasks, 'panda_reload')
    server.register_function(manager.get_sim_root, 'panda_connect')
    server.register_function(manager.get_next_task, 'panda_request')
    server.register_function(manager.set_complete, 'panda_complete')
    server.register_function(manager.set_error, 'panda_error')
    server.register_function(manager.get_status, 'panda_status')
    server.register_multicall_functions()
    server.register_introspection_functions()

    logger.info("Started XML-RPC server at %s:%d" % (host, port))

    server.serve_forever()
