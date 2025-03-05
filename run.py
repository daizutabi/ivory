import polars as pl
from ivory import abs_i64, pig_latinnify

df = pl.DataFrame(
    {
        "a": [1, -1, None],
        "b": [4.1, 5.2, -6.3],
        "c": ["hello", "everybody!", "!"],
    },
)
print(df.with_columns(abs_i64("a").name.suffix("_abs")))


df = pl.DataFrame(
    {
        "english": ["this", "is", "not", "pig", "latin"],
    },
)
result = df.with_columns(pig_latin=pig_latinnify("english"))
print(result)
