import numpy as np
import pytest

from ivory.core.instance import create_instance, instantiate


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


def test_create_instance(params):
    trainer = create_instance(params, "run.trainer")
    assert trainer.max_epochs == 10


def test_instantiate_global():
    a = {"class": "numpy.array", "object": [1, 2, 3]}
    a = instantiate(a, {"c": 0})
    assert a[0] == 1
