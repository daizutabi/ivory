from ivory.core.parser import Parser


def test_parser():
    parser = Parser()
    parser.parse_args(a=1, b="1-2", c="4,5,6")
    assert parser.values[0] == [1]
    assert parser.values[1] == range(1, 3)
    assert parser.values[2] == [4, 5, 6]

    parser.parse_args(["a=1", "b=1-2", "c=4,5,6"], d="x")
    assert parser.values[0] == [1]
    assert parser.values[1] == range(1, 3)
    assert parser.values[2] == [4, 5, 6]
    assert parser.values[3] == ["x"]

    parser.parse_args(dict(a=1, b="1-2", c="4,5,6"))
    assert parser.values[0] == [1]
    assert parser.values[1] == range(1, 3)
    assert parser.values[2] == [4, 5, 6]


def test_parser_mode():
    parser = Parser()
    parser.parse_args(a=1, b="1", c="4")
    assert parser.mode == "single"
    parser.parse_args(a=1, b="1-3", c="4")
    assert parser.mode == "scan"
    parser.parse_args(a=1, b="1-3", c="4,6")
    assert parser.mode == "prod"
