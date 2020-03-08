import numpy as np
import pandas as pd

from ivory.core.instance import instantiate


def test_instantiate_single(config_single):
    obj = instantiate(config_single)
    assert isinstance(obj["data"], np.ndarray)
    assert obj["data"][0] == 1


def test_parse_multi(config):
    obj = instantiate(config)
    assert isinstance(obj["series"], pd.Series)
    assert obj["series"][0] == 1

    config["series"]["data"] = "$.data"
    obj = instantiate(config)
    assert obj["series"][0] == 1
    assert len(obj["series"]) == 2

    config["series"]["data"] = "$.data.shape"
    obj = instantiate(config)
    assert obj["series"][0] == 2
    assert len(obj["series"]) == 1


def test_parse_default(config):
    obj1 = instantiate(config)
    obj2 = instantiate(config)
    assert obj1["data"] is not obj2["data"] and obj1["series"] is not obj2["series"]
    obj2 = instantiate(config, default={"data": obj1["data"]})
    assert obj1["data"] is obj2["data"] and obj1["series"] is not obj2["series"]
    obj2 = instantiate(config, default=obj1)
    assert obj1["data"] is obj2["data"] and obj1["series"] is obj2["series"]


def test_parse_keys(config):
    obj = instantiate(config, names=["data"])
    assert "data" in obj and "series" not in obj


def test_instantiate_extra():
    config = {"data": {"a": 1, "b": 2}, "x": 100}
    obj = instantiate(config)
    assert isinstance(obj["data"], dict)
    assert obj["data"]["a"] == 1 and obj["data"]["b"] == 2
    assert obj["x"] == 100

    config = {"def": "numpy.array", "object": [1, 2]}
    obj = instantiate(config)
    assert isinstance(obj, np.ndarray)
