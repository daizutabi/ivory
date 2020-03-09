import numpy as np
import pandas as pd

from ivory.core.instance import get_classes, instantiate


def test_get_classes(params):
    it = get_classes(params)
    assert next(it) is pd.Series


def test_instantiate_single(params_single):
    obj = instantiate(params_single)
    assert isinstance(obj["data"], np.ndarray)
    assert obj["data"][0] == 1


def test_instantiate_multi(params):
    obj = instantiate(params)
    assert isinstance(obj["series"], pd.Series)
    assert obj["series"][0] == 1

    params["series"]["data"] = "$.data"
    obj = instantiate(params)
    assert obj["series"][0] == 1
    assert len(obj["series"]) == 2

    params["series"]["data"] = "$.data.shape"
    obj = instantiate(params)
    assert obj["series"][0] == 2
    assert len(obj["series"]) == 1


def test_instantiate_default(params):
    obj1 = instantiate(params)
    obj2 = instantiate(params)
    assert obj1["data"] is not obj2["data"] and obj1["series"] is not obj2["series"]
    obj2 = instantiate(params, default={"data": obj1["data"]})
    assert obj1["data"] is obj2["data"] and obj1["series"] is not obj2["series"]
    obj2 = instantiate(params, default=obj1)
    assert obj1["data"] is obj2["data"] and obj1["series"] is obj2["series"]


def test_instantiate_keys(params):
    obj = instantiate(params, names=["data"])
    assert "data" in obj and "series" not in obj


def test_instantiate_extra():
    params = {"data": {"a": 1, "b": 2}, "x": 100}
    obj = instantiate(params)
    assert isinstance(obj["data"], dict)
    assert obj["data"]["a"] == 1 and obj["data"]["b"] == 2
    assert obj["x"] == 100

    params = {"def": "numpy.array", "object": [1, 2]}
    obj = instantiate(params)
    assert isinstance(obj, np.ndarray)
