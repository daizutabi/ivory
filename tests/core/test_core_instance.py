import numpy as np
import pandas as pd
import pytest

from ivory.core.instance import create_instance, instantiate, parse_value


def test_instantiate_single(params_single):
    obj = instantiate(params_single["data"])
    assert isinstance(obj, np.ndarray)
    assert obj[0] == 1


def test_parse_value(params):
    objs = parse_value(params, {}, "")
    assert isinstance(objs["series"], pd.Series)
    assert objs["series"][0] == 1

    params["series"]["data"] = "$.data.shape"
    obj = parse_value(params, {}, "")
    assert obj["series"][0] == 2
    assert len(obj["series"]) == 1


def test_instantiate_extra():
    params = {"call": "numpy.array", "object": [1, 2]}
    obj = instantiate(params)
    assert isinstance(obj, np.ndarray)

    with pytest.raises(ValueError):
        params = {"x": 100, "data": "$.x"}
        instantiate(params)

    with pytest.raises(ValueError):
        params = {"call": "array", "object": [1, 2]}
        instantiate(params)


def test_def():
    params = {"def": "numpy.array", "object": [3, 4]}
    obj = instantiate(params)
    assert callable(obj)
    assert all(obj() == [3, 4])

    params = {"def": "numpy.array"}
    obj = instantiate(params)
    assert obj is np.array


def test_create_instance(params_path):
    a = create_instance("environment.tracker", params_path)
    assert hasattr(a, "tracking_uri")


def test_instantiate_global():
    a = {"class": "numpy.array", "object": [1, 2, 3]}
    a = instantiate(a, {"c": 0})
    assert a[0] == 1
