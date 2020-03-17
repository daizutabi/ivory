import pytest

import ivory


@pytest.fixture
def params_path():
    return "tests/params.yaml"


@pytest.fixture
def params_single():
    return {"data": {"call": "numpy.array", "object": [1, 2]}}


@pytest.fixture
def params():
    return {
        "data": {"call": "numpy.array", "object": [1, 2]},
        "series": {"class": "pandas.Series", "data": "$"},
        "data2": {"call": "numpy.array", "object": "$.data"},
        "data3": {"call": "numpy.array", "object": "$.data.0"},
    }


@pytest.fixture
def environment(params_path):
    return ivory.create_environment(params_path)


@pytest.fixture
def experiment(params_path, environment):
    return environment.create_experiment(params_path)


@pytest.fixture
def run(params_path, experiment):
    return experiment.create_run(params_path)
