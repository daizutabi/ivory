import pytest


@pytest.fixture
def config_single():
    return {"data": {"def": "numpy.array", "object": [1, 2]}}


@pytest.fixture
def config():
    return {
        "data": {"def": "numpy.array", "object": [1, 2]},
        "series": {"class": "pandas.Series", "data": "$"},
    }
