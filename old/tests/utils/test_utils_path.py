from ivory.utils.path import literal_eval, to_uri


def test_to_uri():
    assert to_uri("file:///a/b/c") == "file:///a/b/c"
    assert "~" not in to_uri("~/a/b/c")


def test_literal_eval():
    x = {"a": "1e2", "b": [1, "1e-1", {"c": ["1e3", "abc"]}]}
    y = literal_eval(x)
    assert y == {"a": 100.0, "b": [1, 0.1, {"c": [1000.0, "abc"]}]}
