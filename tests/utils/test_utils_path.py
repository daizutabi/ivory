from ivory.utils.path import to_uri


def test_to_uri():
    assert to_uri("file:///a/b/c") == "file:///a/b/c"
    assert "~" not in to_uri("~/a/b/c")
