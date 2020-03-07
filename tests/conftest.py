import numpy as np
import pytest
from omegaconf import OmegaConf
from pandas import DataFrame

from ivory.utils import kfold_split


@pytest.fixture
def config_single():
    return OmegaConf.create([{"data": {"def": "numpy.array", "object": [1, 2]}}])


@pytest.fixture
def config():
    c = {"data": {"def": "numpy.array", "object": [1, 2]}}
    return OmegaConf.create([c, {"series": {"class": "pandas.Series", "data": "$"}}])


@pytest.fixture(scope="session")
def data():
    num_samples = 1000
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    df = DataFrame(xy, columns=["x", "y"])
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    df["z"] = ((df.x + dx) * (df.y + dy)).astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df
