import os
import shutil
import sys

import pytest
import tensorflow as tf

from ivory.core.client import create_client

sys.path.insert(0, os.path.abspath("tests/examples"))


@pytest.fixture(scope="session")
def setup():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


@pytest.fixture(scope="session")
def client(setup):
    client = create_client(directory="tests/examples")
    yield client
    if os.path.exists("tests/examples/mlruns"):
        shutil.rmtree("tests/examples/mlruns")


@pytest.fixture(scope="session")
def experiment(client):
    return client.create_experiment("example")


@pytest.fixture(scope="function")
def params(experiment):
    yield experiment.create_params()[0]


@pytest.fixture(scope="session")
def tracker(experiment):
    yield experiment.tracker


@pytest.fixture(scope="session")
def run(experiment):
    yield experiment.create_run()


@pytest.fixture(scope="session")
def task(experiment):
    yield experiment.create_task()


@pytest.fixture(scope="session")
def study(experiment):
    yield experiment.create_study()


@pytest.fixture(scope="session")
def objective(study):
    yield study.objective


@pytest.fixture(scope="session")
def data(run):
    yield run.datasets.data


@pytest.fixture(scope="session")
def dataset(run):
    yield run.datasets.dataset


@pytest.fixture(scope="session")
def datasets(run):
    yield run.datasets


@pytest.fixture(scope="session")
def results(run):
    yield run.results


@pytest.fixture(scope="session")
def metrics(run):
    yield run.metrics


@pytest.fixture(scope="session")
def trainer(run):
    yield run.trainer


@pytest.fixture(scope="session")
def tuner(study):
    yield study.tuner
