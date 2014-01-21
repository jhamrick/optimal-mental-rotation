from .util import setup_config

import sys
import os
import subprocess as sp
import pytest
import shutil
from path import path


@pytest.fixture(scope="module")
def config(request):
    tmp_path = path("/tmp/mental_rotation")
    shutil.copytree("./experiment", tmp_path.joinpath("experiment"))

    config_path = tmp_path.joinpath('config.ini')
    config = setup_config(tmp_path)
    with open(config_path, 'w') as fh:
        config.write(fh)

    def fin():
        tmp_path.rmtree_p()
    request.addfinalizer(fin)

    return config_path


def test_convert_old_stimuli(config):
    code = sp.call([
        "./bin/convert_old_stimuli.py", 
        "--from", "old", 
        "--to", "new", 
        "-c", config])
    assert code == 0

    code = sp.call([
        "./bin/convert_old_stimuli.py", 
        "--from", "old", 
        "--to", "new", 
        "-f", 
        "-c", config])
    assert code == 0


def test_experiment_generate_configs(config):
    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", config])
    assert code == 0

    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", config, "-f"])
    assert code == 0


def test_experiment_deploy_experiment(config):
    code = sp.call([
        "./bin/experiment/deploy_experiment.py", 
        "-c", config])
    assert code == 0


def test_pre_experiment(config):
    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", config])
    assert code == 1

    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", config,
        "-a"
    ])
    assert code == 0

    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", config,
        "-a", "-f"
    ])
    assert code == 0


def test_experiment_fetch_data(config):
    pass


def test_experiment_process_data(config):
    pass


def test_experiment_extract_workers(config):
    pass


def test_post_experiment(config):
    pass


def test_model_run_simulations(config):
    pass


def test_model_process_simulations(config):
    pass


def test_model(config):
    pass
