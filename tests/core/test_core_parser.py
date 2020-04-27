from ivory.core import parser


def test_parser():
    args = parser.parse_args(a=1, b="1-2", c="4,5,6")
    assert args['a'] == [1]
    assert args['b'] == range(1, 3)
    assert args['c'] == [4, 5, 6]

    args = parser.parse_args(["a=1", "b=1-2", "c=4,5,6"], d="x")
    assert args['a'] == [1]
    assert args['b'] == range(1, 3)
    assert args['c'] == [4, 5, 6]
    assert args['d'] == ["x"]

    args = parser.parse_args(dict(a=1, b="1-2,4-5", c="4,5,6"))
    assert args['a'] == [1]
    assert args['b'] == [1, 2, 4, 5]
    assert args['c'] == [4, 5, 6]
