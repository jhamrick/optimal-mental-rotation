import signal
import sys
import multiprocessing as mp
import logging
from threading import Thread, current_thread
from utils import parse_address
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


def run_client(**kwargs):
    """Run the simulation client manager."""

    save = kwargs.get("save", False)
    timeout = kwargs.get("timeout", 310)
    address = kwargs.get("address", ("127.0.0.1", 50000))
    authkey = kwargs.get("authkey", None)
    n_procs = kwargs.get("n_procs", mp.cpu_count())
    max_tries = kwargs.get("max_tries", 3)

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
    for ithread in xrange(n_procs):
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


def parse_and_run_client(args):
    kwargs = {
        'save': args.save,
        'timeout': args.timeout,
        'address': args.address,
        'authkey': args.authkey,
        'n_procs': args.num_procs,
        'max_tries': args.max_tries
    }
    run_client(**kwargs)


def create_client_parser(parser):
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save data to disk.")
    parser.add_argument(
        "-T", "--timeout",
        default=310,
        type=int,
        help="Timeout (in seconds) before process restarts.")
    parser.add_argument(
        "-a", "--address",
        default=("127.0.0.1", 50000),
        type=parse_address,
        help="Address (host:port) of server.")
    parser.add_argument(
        "-k", "--authkey",
        default=None,
        help="Server authentication key.")
    parser.add_argument(
        "-n", "--num-processes",
        default=mp.cpu_count(),
        dest="num_procs",
        type=int,
        help="Number of client processes.")
    parser.add_argument(
        "-m", "--max-tries",
        default=3,
        dest="max_tries",
        type=int,
        help="Number of times to try running a task.")
    parser.set_defaults(func=parse_and_run_client)
