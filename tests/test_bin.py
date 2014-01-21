from .util import setup_temp, setup_config

import sys
import os
import subprocess as sp
import pytest


@pytest.fixture(scope="module")
def config(request):
    tmp_path = setup_temp()
    config_path = tmp_path.joinpath('config.ini')
    config = setup_config(tmp_path)
    with open(config_path, 'w') as fh:
        config.write(fh)

    def fin():
        tmp_path.rmtree_p()
    request.addfinalizer(fin)

    return config_path


@pytest.fixture(scope="module")
def config_real_stimuli(request):
    tmp_path = setup_temp()
    config_path = tmp_path.joinpath('config.ini')
    config = setup_config(tmp_path)
    config.set("paths", "stimuli", "stimuli")
    with open(config_path, 'w') as fh:
        config.write(fh)

    def fin():
        tmp_path.rmtree_p()
    request.addfinalizer(fin)

    return config_path


@pytest.fixture(scope="module")
def config_real_experiment(request):
    tmp_path = setup_temp()
    config_path = tmp_path.joinpath('config.ini')
    config = setup_config(tmp_path)
    config.set("paths", "experiment", "experiment")
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


def test_experiment_generate_configs(config_real_stimuli):
    config = config_real_stimuli

    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", config])
    assert code == 0

    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", config, "-f"])
    assert code == 0


def test_experiment_deploy_experiment(tmpdir, config_real_experiment):
    config = config_real_experiment

    pth = tmpdir.mkdir("exp")
    code = sp.call([
        "./bin/experiment/deploy_experiment.py", 
        "-c", config,
        "-H", "localhost",
        "-n",
        pth.strpath
    ])
    assert code == 0

    code = sp.call([
        "./bin/experiment/deploy_experiment.py", 
        "-c", config,
        "-H", "localhost",
        pth.strpath
    ])
    assert code == 0


def test_pre_experiment(config):
    pass


def test_experiment_extract_workers(config):
    pass


def test_experiment_fetch_data(config):
    pass


def test_experiment_process_data(config):
    pass


def test_model_generate_script(config):
    pass


def test_model_process_simulations(config):
    pass


def test_model_run_simulations(config):
    pass


def test_model(config):
    pass


def test_post_experiment(config):
    pass
