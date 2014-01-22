import subprocess as sp
import pytest


def test_convert_old_stimuli(tmp_config):
    code = sp.call([
        "./bin/convert_old_stimuli.py", 
        "--from", "old", 
        "--to", "new", 
        "-c", tmp_config])
    assert code == 0

    code = sp.call([
        "./bin/convert_old_stimuli.py", 
        "--from", "old", 
        "--to", "new", 
        "-f", 
        "-c", tmp_config])
    assert code == 0


@pytest.mark.usefixtures("tmp_experiment")
def test_experiment_generate_configs(tmp_config):
    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", tmp_config])
    assert code == 0

    code = sp.call([
        "./bin/experiment/generate_configs.py", 
        "-c", tmp_config, "-f"])
    assert code == 0


@pytest.mark.usefixtures("tmp_experiment")
def test_experiment_deploy_experiment(tmp_config):
    code = sp.call([
        "./bin/experiment/deploy_experiment.py", 
        "-c", tmp_config])
    assert code == 0


@pytest.mark.usefixtures("tmp_experiment")
def test_pre_experiment(tmp_config):
    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", tmp_config])
    assert code == 1

    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", tmp_config,
        "-a"
    ])
    assert code == 0

    code = sp.call([
        "./bin/pre_experiment.py", 
        "-c", tmp_config,
        "-a", "-f"
    ])
    assert code == 0


def test_experiment_fetch_data(tmp_config):
    pass


def test_experiment_process_data(tmp_config):
    pass


def test_experiment_extract_workers(tmp_config):
    pass


def test_post_experiment(tmp_config):
    pass


def test_model_run_simulations(tmp_config):
    pass


def test_model_process_simulations(tmp_config):
    pass


def test_model(tmp_config):
    pass
