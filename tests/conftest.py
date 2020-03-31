import copy
import os
import shutil
import sys

import pytest

from ivory.core.client import create_client

sys.path.insert(0, os.path.abspath("tests"))


@pytest.fixture(scope="session")
def client():
    client = create_client(directory="tests")
    client.create_experiment("example")
    yield client
    if os.path.exists("tests/mlruns"):
        shutil.rmtree("tests/mlruns")


@pytest.fixture(scope="function")
def params(client):
    yield copy.deepcopy(client.params)


@pytest.fixture(scope="session")
def experiment(client):
    yield client.experiment


@pytest.fixture(scope="session")
def tracker(experiment):
    yield experiment.tracker


@pytest.fixture(scope="session")
def data(experiment):
    yield experiment.data


@pytest.fixture(scope="session")
def objective(experiment):
    yield experiment.objective


@pytest.fixture(scope="session")
def run(client):
    yield client.create_run()


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
