import sys
import multiprocessing as mp
import logging
from multiprocessing.managers import BaseManager
from datetime import datetime, timedelta
from path import path
from tasks import Tasks

logger = logging.getLogger("mental_rotation.sims.server")


class ServerManager(BaseManager):

    def add_tasks(self, force):
        params = self.get_params()

        tasks_file = path(params["tasks_path"])
        completed_file = tasks_file.dirname().joinpath("completed.json")
        if tasks_file.exists() and not force:
            tasks = Tasks.load(tasks_file)
            completed = Tasks.load(completed_file)
        else:
            tasks, completed = Tasks.create(params)
            tasks.save(tasks_file)
            completed.save(completed_file)

        task_queue = self.get_task_queue()
        added_tasks = Tasks()
        for task_name in sorted(tasks.keys()):
            task = tasks[task_name]
            complete = completed[task_name]
            if force or not complete:
                task_queue.put(task)
                added_tasks[task_name] = task

        logger.info("%d tasks queued", len(added_tasks))
        return added_tasks

    def manage_tasks(self, tasks):
        """Receives completed tasks from the clients and updates the tasks
        file.

        """

        start_time = datetime.now()
        params = self.get_params()

        tasks_file = path(params["tasks_path"])
        completed_file = tasks_file.dirname().joinpath("completed.json")
        num_tasks = len(tasks)

        task_queue = self.get_task_queue()
        done_queue = self.get_done_queue()

        try:
            num_remaining = task_queue.qsize()
        except NotImplementedError:
            num_remaining = num_tasks
        num_processed = 0

        def running():
            return (not task_queue.empty() or
                    task_queue.join() or
                    not done_queue.empty())

        # Main task loop. Stopping criteria: no more tasks and all
        # clients have signaled task done.
        while running():
            # Wait for a done task to arrive.
            task_name = done_queue.get()
            completed = Tasks.load(completed_file)
            completed[task_name] = True

            # Save task to disk.
            completed.save(completed_file)

            # Report progress.
            num_processed += 1
            try:
                num_remaining = task_queue.qsize()
            except NotImplementedError:
                num_remaining -= 1
            num_finished = num_tasks - num_remaining
            progress = 100 * float(num_finished) / num_tasks
            dt = datetime.now() - start_time
            avg_dt = timedelta(
                seconds=(dt.total_seconds() / float(num_processed + 1e-5)))
            time_left = timedelta(
                seconds=(avg_dt.total_seconds() * num_remaining))

            logger.info("-" * 60)
            logger.info("Task `%s` complete", task_name)
            logger.info("Progress: %d/%d (%.2f%%)",
                        num_finished, num_tasks, progress)
            logger.info("Time elapsed  : %s", str(dt))
            logger.info("Time per task : %s", str(avg_dt))
            logger.info("Time remaining: %s", str(time_left))

    @classmethod
    def register_shared(cls, with_callable, params=None):
        if with_callable:
            task_queue = mp.JoinableQueue()
            done_queue = mp.Queue()

            cls.register(
                "get_params", lambda: params,
                proxytype=mp.managers.DictProxy)
            cls.register(
                "get_task_queue", lambda: task_queue)
            cls.register(
                "get_done_queue", lambda: done_queue)

            logger.info("Registered shared resources")

        else:
            cls.register("get_params")
            cls.register("get_task_queue")
            cls.register("get_done_queue")


def run_server(params, address, authkey, force, loglevel):
    """Run the simulation server manager."""

    mplogger = mp.log_to_stderr()
    mplogger.setLevel(loglevel)
    logger.setLevel(loglevel)

    # Set up parameters and task data.
    ServerManager.register_shared(with_callable=True, params=params)
    logger.info("Registered shared resources")

    # Start the server and set the shared data.
    mgr = ServerManager(address=address, authkey=authkey)
    mgr.start()
    tasks = mgr.add_tasks(force)

    # Monitor the tasks and update the tasks file as completed tasks
    # arrive from the clients.
    mgr.manage_tasks(tasks)

    logger.info("Jobs complete. Shutting down.")
    mgr.shutdown()
    sys.exit(0)
