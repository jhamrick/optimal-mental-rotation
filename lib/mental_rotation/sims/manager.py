import logging
import json
import multiprocessing as mp
import Queue

from path import path
from datetime import datetime, timedelta
from SimpleXMLRPCServer import SimpleXMLRPCServer

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

        self.create_tasks()
        self.load_tasks()

        self.start_time = datetime.now()
        self.num_finished = 0
        self.errors = []

    def create_tasks(self):
        logger.debug("Creating tasks...")

        tasks_file = path(self.params["tasks_path"])
        completed_file = path(self.params["completed_path"])

        if tasks_file.exists() and completed_file.exists() and not self.force:
            logger.debug("Note: tasks file already exists")

        else:
            tasks, completed = Tasks.create(self.params)
            tasks.save(tasks_file)
            completed.save(completed_file)

            logger.debug("Tasks file created")

    def load_tasks(self):
        logger.debug("Loading tasks...")

        tasks_file = path(self.params["tasks_path"])
        completed_file = path(self.params["completed_path"])

        self.tasks = Tasks.load(tasks_file)
        self.completed = Tasks.load(completed_file)

        logger.info("%d tasks loaded", len(self.tasks))

        self.num_tasks = 0
        for task_name in sorted(self.tasks.keys()):
            complete = self.completed[task_name]
            if not complete:
                self.queue.put(task_name)
                self.num_tasks += 1

        logger.info("%d tasks queued", self.num_tasks)

    def get_sim_root(self):
        logger.debug("Got request for sim_root")

        # send the client the path to save the data
        return str(path(self.params['sim_root']).abspath())

    def get_next_task(self):
        # process a request for a new task
        try:
            task_name = self.queue.get(False)
        except Queue.Empty:
            return None

        task = self.tasks[task_name]
        if self.completed[task_name]:
            raise RuntimeError(
                "task '%s' has already been completed!" % task_name)

        logger.debug("Allocating task '%s'", task_name)
        return json.dumps(task)

    def extract_data(self, task_name, zip_path):
        # extract the tar archive that was sent over
        task = self.tasks[task_name]
        data_path = path(task['data_path'])
        zip_path = path(zip_path)
        if not zip_path.exists():
            logger.error("Data not found for task '%s'", task_name)
            return

        if data_path.exists():
            data_path.rmtree_p()

        logger.debug("Extracting data for task '%s'", task_name)
        cmd = ['tar', '-xf', zip_path, '-C', data_path.dirname()]
        run_command(logger, cmd)

        zip_path.remove()

    def set_complete(self, task_name):
        if self.completed[task_name]:
            raise RuntimeError(
                "task '%s' has already been completed!" % task_name)

        # mark the task as done
        completed_file = path(self.params["completed_path"])
        self.completed[task_name] = True
        self.completed.save(completed_file)

        # report the progress
        self.num_finished += 1
        self.report(task_name)

    def set_error(self, task_name):
        logger.error("Task '%s' failed with an error", task_name)
        self.errors.append(task_name)

    def report(self, task_name):
        progress = 100 * float(self.num_finished) / self.num_tasks
        dt = datetime.now() - self.start_time
        avg_dt = timedelta(
            seconds=(dt.total_seconds() / float(self.num_finished + 1e-5)))
        time_left = timedelta(
            seconds=(avg_dt.total_seconds() * (
                self.num_tasks - self.num_finished)))

        logger.info("=" * 40)
        logger.info("Task `%s` complete", task_name)
        logger.info("Progress: %d/%d (%.2f%%)",
                    self.num_finished, self.num_tasks, progress)
        logger.info("Time elapsed  : %s", str(dt))
        logger.info("Time per task : %s", str(avg_dt))
        logger.info("Time remaining: %s", str(time_left))
        logger.info("-" * 40)


class TaskManagerServer(SimpleXMLRPCServer):

    def __init__(self, manager, host="127.0.0.1", port=55556):
        SimpleXMLRPCServer.__init__(
            self, (host, port), logRequests=False, allow_none=True)

        self.register_function(manager.load_tasks, 'panda_reload')
        self.register_function(manager.get_sim_root, 'panda_connect')
        self.register_function(manager.get_next_task, 'panda_request')
        self.register_function(manager.extract_data, 'panda_extract')
        self.register_function(manager.set_complete, 'panda_complete')
        self.register_function(manager.set_error, 'panda_error')

        self.register_multicall_functions()
        self.register_introspection_functions()

        logger.info("Started XML-RPC server at %s:%d" % (host, port))


def run(params, force):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(params['loglevel'])
    logger.setLevel(params['loglevel'])

    manager = TaskManager(params, force)
    server = TaskManagerServer(manager)
    server.serve_forever()