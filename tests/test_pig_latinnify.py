import polars as pl


def test_piglatinnify():
    from ivory import pig_latinnify

    df = pl.DataFrame({"english": ["this", "is", "not", "pig", "latin"]})
    result = df.select(pig_latinnify("english").alias("pig_latin"))

    expected_df = pl.DataFrame(
        {"pig_latin": ["histay", "siay", "otnay", "igpay", "atinlay"]},
    )

    assert result.equals(expected_df)
