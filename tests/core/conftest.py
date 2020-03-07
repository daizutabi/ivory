import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config_single():
    return OmegaConf.create([{"data": {"def": "numpy.array", "object": [1, 2]}}])


@pytest.fixture
def config():
    c = {"data": {"def": "numpy.array", "object": [1, 2]}}
    return OmegaConf.create([c, {"series": {"class": "pandas.Series", "data": "$"}}])
