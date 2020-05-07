from ivory.utils.range import Range


def test_range():
    x = Range(1, 2)
    assert x.is_integer
    assert not x.is_float


def test_range_repr():
    x = Range(1, 2)
    assert repr(x) == "Range(1, 2)"
    x = Range(1, 5, 2)
    assert repr(x) == "Range(1, 5, 2)"
    x = Range(1, 5, n=10)
    assert repr(x) == "Range(1, 5, n=10)"
    x = Range(1, 5, 2, n=10)
    assert repr(x) == "Range(1, 5, 2, n=10)"


def test_range_iter():
    x = list(Range(1, 10))
    assert x[0] == 1
    assert x[-1] == 10

    x = list(Range(10, 1))
    assert x[0] == 10
    assert x[-1] == 1

    x = list(Range(0, 8, 3))
    assert x == [0, 3, 6]

    x = list(Range(0, 8, n=4))
    assert x == [0, 3, 5, 8]

    x = list(Range(0.0, 10.0))
    assert len(x) == 11

    x = list(Range(0.0, 10.0, n=5))
    assert len(x) == 5
