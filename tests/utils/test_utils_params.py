import pytest

from ivory.utils.params import colon_to_list, update_dict


def test_update_dict():
    x = {"a": 1, "b": {"x": "abc", "y": 2}, "c": {"z": [0, 1, 2]}}
    update_dict(x, {"a": 2})
    assert x == {"a": 2, "b": {"x": "abc", "y": 2}, "c": {"z": [0, 1, 2]}}
    update_dict(x, {"d": 2})  # add non-existing key
    assert x == {"a": 2, "b": {"x": "abc", "y": 2}, "c": {"z": [0, 1, 2]}, "d": 2}
    x.pop("d")
    update_dict(x, {"b.x": "def"})
    assert x == {"a": 2, "b": {"x": "def", "y": 2}, "c": {"z": [0, 1, 2]}}
    update_dict(x, {"c.z": [3, 4]})
    assert x == {"a": 2, "b": {"x": "def", "y": 2}, "c": {"z": [3, 4]}}
    update_dict(x, {"b.y": 5})
    assert x == {"a": 2, "b": {"x": "def", "y": 5}, "c": {"z": [3, 4]}}
    with pytest.raises(ValueError):
        update_dict(x, {"c.z": 3})

    x = {"a": 1, "b": {"x": "abc", "y": 2, "z": [0, 1, 2]}}
    update_dict(x, {"b": {"z": [0]}, "b.x": "def"})
    assert x == {"a": 1, "b": {"x": "def", "y": 2, "z": [0]}}


def test_dot_to_list():
    x = {"a": 1, "b.c": 2, "c:0": 3, "c:1": 4, "d.x:0": 10, "d.x:1": 20}
    y = colon_to_list(x)
    assert y == {"a": 1, "b.c": 2, "c": [3, 4], "d.x": [10, 20]}

    with pytest.raises(KeyError):
        colon_to_list({"a:1": 3, "a:0": 5})
    with pytest.raises(KeyError):
        colon_to_list({"a": 3, "a:0": 5})
