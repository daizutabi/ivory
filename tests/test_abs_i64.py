import polars as pl


def test_abs_i64():
    from ivory import abs_i64

    df = pl.DataFrame({"a": [1, -1, None]})
    result = df.select(abs_a=abs_i64("a"))
    expected_df = pl.DataFrame({"abs_a": [1, 1, None]})

    assert result.equals(expected_df)
