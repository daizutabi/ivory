import os
import shutil
import sys

import pytest

from ivory.core.client import create_client

sys.path.insert(0, os.path.abspath("examples"))


@pytest.fixture(scope="session")
def client():
    client = create_client(directory="examples")
    yield client
    if os.path.exists("examples/mlruns"):
        shutil.rmtree("examples/mlruns")


@pytest.fixture(scope="session")
def experiment(client):
    return client.create_experiment("example")


@pytest.fixture(scope="session")
def evaluator(client):
    return client.create_evaluator()


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
    yield run.dataloaders.data


@pytest.fixture(scope="session")
def dataset(run):
    yield run.dataloaders.dataset


@pytest.fixture(scope="session")
def dataloaders(run):
    yield run.dataloaders


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
