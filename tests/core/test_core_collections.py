import pytest

from ivory.core.collections import Dict, List, Missing


def test_dict():
    a = Dict()
    a["x"] = 1
    assert a.x == 1

    a = Dict()
    a.set(x=1, y=2)
    assert a.x == 1
    assert a.y == 2

    assert "x" in a
    assert list(a.keys()) == ["x", "y"]
    assert list(a.values()) == [1, 2]
    assert "x" in dir(a)

    with pytest.raises(KeyError):
        a["z"]

    assert isinstance(a.z, Missing)

    with pytest.raises(AttributeError):
        a.z.abc
    with pytest.raises(AttributeError):
        a.z["abc"]


def test_list():
    a = List()
    assert a

    a.set([1, 2, 3])
    assert len(a) == 3
    a.append(4)
    assert len(a) == 4
    assert a.copy() == [1, 2, 3, 4]
    assert a[3] == 4
    assert 4 in a
    for k in a:
        pass
    assert k == 4
    assert repr(a) == "List([1, 2, 3, 4])"


def test_missing():
    missing = Missing("abc", "def")
    assert not missing
