from .util import setup_temp, setup_config

import sys
import os
import subprocess as sp
import pytest


@pytest.fixture(scope="module")
def config(request):
    tmp_path = setup_temp()
    config_path = setup_config(tmp_path)
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
