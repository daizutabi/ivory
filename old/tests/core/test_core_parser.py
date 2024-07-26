import pytest

from ivory.core import parser
from ivory.utils.range import Range


def test_parser():
    args = parser.parse_args(a=[1, 2])
    assert args == {"a": [1, 2]}

    args = parser.parse_args(a=1, b="1-2", c="4,5,6")
    assert args["a"] == [1]
    assert isinstance(args["b"], Range)
    assert args["c"] == [4, 5, 6]

    args = parser.parse_args(["a=1", "b=1-2", "c=4,5,6"], d="x")
    assert args["a"] == [1]
    assert args["c"] == [4, 5, 6]
    assert args["d"] == ["x"]

    args = parser.parse_args(dict(a=1, b="1-2,4-5", c="4,5,6"))
    assert args["a"] == [1]
    assert args["b"] == [1, 2, 4, 5]
    assert args["c"] == [4, 5, 6]

    with pytest.raises(ValueError):
        parser.parse_args(1)

    args = parser.parse_args(["a="])
    assert args["a"] == [""]

    x = parser.parse_args(["a=4-0"])["a"]
    assert x.start == 4
    assert x.stop == 0
    assert x.step == 1
    assert x.num == 0
    assert len(x) == 5

    x = parser.parse_args(["a=0.0-1.0:100"])["a"]
    assert x.start == 0.0
    assert x.stop == 1.0
    assert x.step == 1
    assert x.num == 100
    assert len(x) == 100

    x = parser.parse_args(["a=0-10-3"])["a"]
    assert x.start == 0
    assert x.stop == 10
    assert x.step == 3

    x = parser.parse_args(["a=x-y"])["a"]
    assert x == ["x-y"]
