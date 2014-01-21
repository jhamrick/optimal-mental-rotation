import signal
import sys
import multiprocessing as mp
import logging
from threading import Thread, current_thread
from server import ServerManager
from simulation import Simulation

logger = logging.getLogger("mental_rotation.sims.client")

def signal_handler(signal, frame):
    mp.util.debug("Keyboard interrupt!")
    sys.exit(100)

signal.signal(signal.SIGINT, signal_handler)


def worker_thread(mgr, info_lock, save=False, max_tries=3, timeout=1e5):

    params = mgr.get_params()
    task_queue = mgr.get_task_queue()
    done_queue = mgr.get_done_queue()

    def retry(task):
        task_queue.put(task)
        task_queue.task_done()

    def finish(task):
        done_queue.put(task["task_name"])
        task_queue.task_done()

    while True:
        try:
            if task_queue.empty():
                break
        except (EOFError, IOError):
            break

        task = task_queue.get()
        task_name = task['task_name']
        job = Simulation(task, params, info_lock, save=save)
        logger.info("Starting task '%s' (%s)", task_name, job.name)
        job.start()
        job.join(timeout=timeout)

        # the thread timed out
        if job.is_alive():
            logger.warning("Timeout for task '%s' (%s)", task_name, job.name)
            job.terminate()
            job.join()
            error = True

        elif job.exitcode == 100:
            logger.warning("Process interrupted, exiting.")
            retry(task)
            break

        # there was an error
        elif job.exitcode != 0:
            logger.error("Task '%s' (%s) exited with code %d",
                         task_name, job.name, job.exitcode)
            error = True

        else:
            logger.info("Task '%s' (%s) complete", task_name, job.name)
            error = False

        if error:
            task["num_tries"] += 1
            if task["num_tries"] >= max_tries:
                logger.error("%d failed attempts at task '%s'",
                             task["num_tries"], task_name)
                task_queue.task_done()
                break

            else:
                logger.warning("Retrying task '%s' (%d/%d)",
                               task_name, task["num_tries"], max_tries)
                retry(task)

        else:
            finish(task)

    logger.info("Ending thread: %s", current_thread())
    sys.exit(0)


def run_client(save, timeout, address, authkey, num_procs, max_tries, loglevel):
    """Run the simulation client manager."""

    mplogger = mp.log_to_stderr()
    mplogger.setLevel(loglevel)
    logger.setLevel(loglevel)

    # Job-specific parameters.
    kwargs = dict(save=save)

    # Connect to the server manager
    ServerManager.register_shared(with_callable=False)
    mgr = ServerManager(address=address, authkey=authkey)
    mgr.connect()

    # create a lock for printing information
    info_lock = mp.Lock()

    # Set up params and task parameters.
    worker_kwargs = {
        'timeout': timeout,
        'save': save,
        'max_tries': max_tries
    }

    logger.info("Starting processes...")
    threads = []
    for ithread in xrange(num_procs):
        thread = Thread(
            name="Thread_%02d" % ithread,
            target=worker_thread,
            args=(mgr, info_lock),
            kwargs=worker_kwargs)
        threads.append(thread)
        thread.start()
        thread.join(timeout=0.2)

    for thread in threads:
        thread.join()

    logger.info("Jobs complete.")
    sys.exit(0)
