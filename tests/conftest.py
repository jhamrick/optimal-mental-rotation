import pytest
import numpy as np
import shutil
import tempfile

from path import path
from ConfigParser import RawConfigParser
from itertools import product

from mental_rotation.stimulus import Stimulus2D

S_sigma = 0.15
step = 0.5
prior = 0.5


@pytest.fixture(scope="session")
def tmproot(request):
    dirpath = path(tempfile.mkdtemp())

    def fin():
        dirpath.rmtree_p()
    request.addfinalizer(fin)

    return dirpath


@pytest.fixture(scope="session")
def config(request, tmproot):
    config = RawConfigParser()

    config.add_section('global')
    config.set('global', 'loglevel', 'INFO')
    config.set('global', 'version', 'test')
    config.set('global', 'seed', '23480')

    config.add_section('paths')
    config.set('paths', 'stimuli', 'tests/stimuli')
    config.set('paths', 'data', tmproot.joinpath('data'))
    config.set('paths', 'figures', tmproot.joinpath('figures'))
    config.set('paths', 'experiment', tmproot.joinpath('experiment'))
    config.set('paths', 'simulations', tmproot.joinpath('data/sim-raw'))
    config.set('paths', 'resources', tmproot.joinpath('resources'))

    config.add_section('experiment')
    config.set('experiment', 'deploy_path', tmproot.joinpath('deploy'))
    config.set('experiment', 'fetch_path', 'http://localhost:22361/data')

    config.add_section('model')
    config.set('model', 'S_sigma', str(S_sigma))
    config.set('model', 'step', step)
    config.set('model', 'prior', prior)

    return config


@pytest.fixture(scope="session")
def tmp_config(request, config, tmproot):
    config_path = tmproot.joinpath('config.ini')

    with open(config_path, 'w') as fh:
        config.write(fh)

    def fin():
        config_path.remove()
    request.addfinalizer(fin)

    return config_path


@pytest.fixture(scope="session")
def tmp_experiment(request, config):
    tmp_path = config.get("paths", "experiment")
    shutil.copytree("./experiment", tmp_path, symlinks=True)

    def fin():
        tmp_path.rmtree_p()
    request.addfinalizer(fin)


@pytest.fixture(autouse=True)
def seed(config):
    seed = config.getint("global", "seed")
    np.random.randint(seed)


@pytest.fixture
def task():
    task = 'task_1'
    return task


@pytest.fixture
def part():
    part = 'part_1'
    return part


@pytest.fixture
def X0():
    X = Stimulus2D.random(8)
    return X


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.function, "full"):
        theta = sorted(range(0, 360, 45) + [39])
        flipped = [True, False]
    elif hasattr(metafunc.function, "once"):
        theta = [0]
        flipped = [False]
    elif hasattr(metafunc.function, "smallrot"):
        theta = [20, 340]
        flipped = [True, False]
    else:
        theta = [39]
        flipped = [True, False]

    all_argnames = ['theta', 'flipped', 'Xa', 'Xb', 'model']
    argnames = [x for x in metafunc.fixturenames if x in all_argnames]

    if len(argnames) == 0:
        return

    argvalues = []

    for t, f in product(theta, flipped):
        # create stimulus
        X = Stimulus2D.random(8)
        if f:
            X.flip([0, 1])
        X.rotate(t)
        Xa = X.copy_from_initial()
        Xb = X.copy_from_vertices()

        # create model
        model = metafunc.cls.cls(
            Xa.vertices, Xb.vertices,
            S_sigma=S_sigma,
            step=step,
            prior=prior)

        args = dict(theta=t, flipped=f, Xa=Xa, Xb=Xb, model=model)
        argvalues.append([args[x] for x in argnames])

    metafunc.parametrize(argnames, argvalues)
