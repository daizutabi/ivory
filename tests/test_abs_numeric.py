import polars as pl


def test_abs_numeric():
    from ivory import abs_numeric

    df = pl.DataFrame({"a": [1, -1, None], "b": [1.0, -1.0, None]})
    result = df.select(abs_numeric("a"), abs_numeric("b"))
    expected_df = pl.DataFrame({"a": [1, 1, None], "b": [1.0, 1.0, None]})

    assert result.equals(expected_df)
