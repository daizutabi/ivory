import pytest


@pytest.fixture
def params_single():
    return {"data": {"def": "numpy.array", "object": [1, 2]}}


@pytest.fixture
def params():
    return {
        "data": {"def": "numpy.array", "object": [1, 2]},
        "series": {"class": "pandas.Series", "data": "$"},
    }
