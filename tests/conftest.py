import copy
import os
import shutil
import sys

import pytest

from ivory.core.client import Client

sys.path.insert(0, os.path.abspath("tests"))


@pytest.fixture(scope="session")
def client():
    yield Client("tests/params.yaml")
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")


@pytest.fixture(scope="function")
def params(client):
    yield copy.deepcopy(client.params)


@pytest.fixture(scope="session")
def experiment(client):
    yield client.experiment


@pytest.fixture(scope="session")
def data(experiment):
    yield experiment.data


@pytest.fixture(scope="session")
def run(client):
    yield client.create_run()


@pytest.fixture(scope="session")
def dataloader(run):
    yield run.dataloader


@pytest.fixture(scope="session")
def metrics(run):
    yield run.metrics


@pytest.fixture(scope="session")
def trainer(run):
    yield run.metrics
