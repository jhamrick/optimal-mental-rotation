import pytest
import multiprocessing as mp
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Lock

from path import path

from mental_rotation.sims.tasks import Tasks
from mental_rotation.sims.manager import TaskManager


class TestTaskManager(object):

    def test_init(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        assert manager.params == sim_params
        assert manager.force is True
        assert manager.tasks is not None
        assert set(manager.completed.keys()) == set(manager.tasks.keys())
        assert isinstance(manager.queue, Queue)
        assert isinstance(manager.info_lock, Lock)
        assert isinstance(manager.save_lock, Lock)

    def test_create_tasks(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.create_tasks()

        # check that files are saved out
        assert path(sim_params['tasks_path']).exists()
        assert path(sim_params['completed_path']).exists()

        # check that save_lock is unlocked
        with pytest.raises(ValueError):
            assert manager.save_lock.release()

        # TODO:  check logger debug worked if file already exists?
        manager.force = False
        manager.create_tasks()

    def test_load_tasks(self, sim_params):
        manager = TaskManager(sim_params, force=True)

        tasks_file = path(sim_params["tasks_path"])
        completed_file = path(sim_params["completed_path"])
        assert tasks_file.exists()
        assert completed_file.exists()

        manager.load_tasks()
        assert manager.tasks == Tasks.load(tasks_file)
        assert manager.completed == Tasks.load(completed_file)

        # check that save_lock is unlocked
        with pytest.raises(ValueError):
            assert manager.save_lock.release()

        # check queue and completed are the right sizes
        total = manager.num_tasks + manager.num_finished
        assert len(manager.tasks.keys()) == total

    def test_get_sim_root(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        assert manager.get_sim_root() == sim_params['sim_root']

    def test_get_next_task(self, sim_params):
        manager = TaskManager(sim_params, force=True)

        manager.get_next_task()

        # check that getting completed task raises error
        for key in manager.completed.keys():
            manager.completed[key] = True
        with pytest.raises(RuntimeError):
            manager.get_next_task()

        # check that if Queue is empty returns None
        manager.queue = mp.Queue()
        assert manager.get_next_task() is None

    def test_set_complete(self, sim_params):
        manager = TaskManager(sim_params, force=True)

        count = manager.num_finished
        total_count = manager.total_finished

        manager.set_complete("example_320_0~0")
        assert manager.completed["example_320_0~0"] is True
        assert manager.num_finished == count + 1
        assert manager.total_finished == total_count + 1

        # check that completed file is saved out
        assert path(sim_params['completed_path']).exists()

        # check that save_lock is unlocked
        with pytest.raises(ValueError):
            assert manager.save_lock.release()

    def test_set_complete_completed(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.set_complete("example_320_0~0")

        # check that already completed task raises error
        with pytest.raises(RuntimeError):
            manager.set_complete("example_320_0~0")

    def test_set_error(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.set_error("example_320_0~0")
        assert "example_320_0~0" in manager.errors

    def test_report(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.report("example_320_0~0")

        # check that info_lock is unlocked
        with pytest.raises(ValueError):
            assert manager.info_lock.release()
