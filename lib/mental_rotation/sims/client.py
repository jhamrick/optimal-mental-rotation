import json
import logging
import multiprocessing as mp
import numpy as np
import signal
import socket
import sys
import tempfile
import time
import traceback

from path import path

import mental_rotation.model as m
from mental_rotation.stimulus import Stimulus2D
from .util import run_command

logger = logging.getLogger("mental_rotation.sims.client")


def signal_handler(signal, frame):
    mp.util.debug("Keyboard interrupt!")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)


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

    error = None
    for isamp in samples:
        dest = data_path.joinpath("sample_%02d" % isamp)
        model = model_class(Xa, Xb, **model_opts)
        try:
            model.sample()
        except:
            error = traceback.format_exc()
            break

        if dest.exists():
            dest.rmtree_p()
        model.save(dest)

    return error


def ssh_copy(source, dest):
    cmd = ['scp', '-q', str(source), str(dest)]
    run_command(logger, cmd)


def connect(host, port):
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s and s.connect_ex((host, port)) == 0:
            if s.send('panda_connect\n') != 0:
                print "sent request"
                sim_root = s.recv(2048)
                if sim_root:
                    s.close()
                    return sim_root
            s.close()

        print ('connect failed, retrying')
        time.sleep(30)


def get_id(host, port):
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s and s.connect_ex((host, port)) == 0:
            if s.send('panda_request\n') != 0:
                print "sent request"
                data = s.recv(2048)
                if data:
                    s.close()
                    task = json.loads(data)
                    return task
            s.close()

        print ('get_id failed, retrying')
        time.sleep(30)


def report(id, msg, host, port):
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s and s.connect_ex((host, port)) == 0:
            if s.send('panda_%s%s\n' % (msg, id)) != 0:
                s.close()
                return
            s.close()

        print ('report failed, retrying')
        time.sleep(30)


def send_back(task, sim_root, host):
    task_name = task["task_name"]
    data_path = path(task["data_path"])
    src_path = data_path.dirname().joinpath("%s.tar.gz" % task_name)

    cmd = ['tar', 'czf', src_path, '-C', data_path.dirname(), data_path.name]
    run_command(logger, cmd)

    if host in ('localhost', '127.0.0.1'):
        dst_path = "%s/%s.tar.gz" % (sim_root, task_name)
    else:
        user = "ubuntu"
        dst_path = "%s@%s:%s/%s.tar.gz" % (user, host, sim_root, task_name)
    ssh_copy(src_path, dst_path)


def worker_job(host, port):

    print "connecting to %s:%s" % (host, port)
    sim_root = connect(host, port)
    tmpdir = path(tempfile.mkdtemp())

    while True:
        # get the next task from the server
        task = get_id(host, port)

        # no more tasks left
        if task == 'no_panda':
            print "no more tasks"
            break

        task_name = task["task_name"]
        print('executing ' + task_name)
        task['data_path'] = tmpdir.joinpath(task_name)
        error = simulate(task)

        if error is not None:
            logger.error("Task '%s' failed with error:\n%s", task_name, error)
            report(task_name, "error", host, port)

        else:
            send_back(task, sim_root, host)
            report(task_name, "complete", host, port)
            print ('done')

        task['data_path'].rmtree_p()

    tmpdir.rmtree_p()


def run(host, port, loglevel):
    # configure logging
    mplogger = mp.log_to_stderr()
    mplogger.setLevel(loglevel)
    logger.setLevel(loglevel)

    # create the pool of worker processes
    #pool = mp.Pool()
    #for i in xrange(mp.cpu_count()):
    #pool.apply_async(worker_job, args=(host, port))
    worker_job(host, port)

    #pool.close()
    #pool.join()
