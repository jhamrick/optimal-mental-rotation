import pytest
import multiprocessing as mp

from path import path

from mental_rotation.sims.manager import TaskManager


class TestTaskManager(object):

    def test_init(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        assert manager.params == sim_params
        assert manager.force is True
        assert manager.tasks is None
        assert manager.completed is None
        assert type(manager.queue) is mp.Queue
        assert type(manager.info_lock) is mp.Lock
        assert type(manager.save_lock) is mp.Lock

    # TODO: save and load tests, check that things are where you expect
    # def test_create_tasks(self):
    # def test_load_tasks(self):

    def test_get_sim_root(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        assert manager.get_sim_root() == sim_params['sim_root']

    def test_get_next_task(self):
        return  # TODO

    def test_set_complete(self, sim_params):
        manager = TaskManager(sim_params, force=True)

        count = manager.num_finished
        total_count = manager.total_finished

        manager.set_complete("test_task")
        assert manager.completed["test_task"] is True
        assert manager.num_finished == count + 1
        assert manager.total_finished == total_count + 1

        # check that completed file is saved out
        assert path(sim_params['completed_path']).exists()

        # check that save_lock is unlocked
        with pytest.raises(mp.ThreadError):
            assert manager.save_lock.release()

    def test_set_complete_completed(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.set_complete("test_task")

        # check that already completed task raises error
        with pytest.raises(RuntimeError):
            manager.set_complete("test_task")

    def test_set_error(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.set_error("test_task")
        assert "test_task" in manager.errors

    def test_report(self, sim_params):
        manager = TaskManager(sim_params, force=True)
        manager.report("test_task")

        # check that info_lock is unlocked
        with pytest.raises(mp.ThreadError):
            assert manager.info_lock.release()
