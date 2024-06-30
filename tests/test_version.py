def test_version():
    from ivory.__about__ import __version__

    assert __version__.count(".") == 2
