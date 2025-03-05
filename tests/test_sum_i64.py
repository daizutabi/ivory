import polars as pl


def test_sum_i64():
    from ivory import sum_i64

    df = pl.DataFrame({"a": [1, None, None], "b": [1, None, None]})
    result = df.select(sum_i64("a", "b").alias("c"))
    expected_df = pl.DataFrame({"c": [2, None, None]})

    assert result.equals(expected_df)
