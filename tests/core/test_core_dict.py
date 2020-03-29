import pytest

from ivory.core.dict import Dict, Missing


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

    assert isinstance(a["z"], Missing)
    assert isinstance(a.z, Missing)

    with pytest.raises(AttributeError):
        a.z.abc
    with pytest.raises(AttributeError):
        a.z["abc"]
