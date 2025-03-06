import polars as pl


def test_cum_sum():
    from ivory import cum_sum

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, None, 5],
            "b": [1, 1, 1, 2, 2, 2],
        },
    )
    result = df.select(cum_sum("a").over("b"))
    expected_df = pl.DataFrame({"a": [1, 3, 6, 4, None, 9]})

    assert result.equals(expected_df)
