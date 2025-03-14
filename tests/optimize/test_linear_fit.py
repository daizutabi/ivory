import numpy as np
import pytest
from polars import DataFrame, Float64, Series
from scipy.optimize import curve_fit


@pytest.fixture(
    scope="module",
    params=[
        ([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]),
        ([1, 2, 3, 1, 2, 3], [4, 3, 2, 7, 6, 5]),
    ],
)
def inputs(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def params(inputs):
    x, y = inputs
    params, _ = curve_fit(lambda x, a, b: a * x + b, x, y)
    return params


@pytest.fixture
def df(inputs):
    x, y = inputs
    return DataFrame({"x": x, "y": y}, schema={"x": Float64, "y": Float64})


def test_linear_fit(df: DataFrame, params):
    from ivory.optimize import linear_fit

    df = df.select(linear_fit("x", "y").struct.unnest())

    x = df["slope"].to_list()[0]
    np.testing.assert_allclose(x, params[0], rtol=1e-5)
    x = df["intercept"].to_list()[0]
    np.testing.assert_allclose(x, params[1], rtol=1e-5)


def test_linear_fit_one():
    from ivory.optimize import linear_fit

    df = DataFrame({"x": [1.0], "y": [2.0]})
    df = df.select(linear_fit("x", "y").struct.unnest())
    assert df["slope"].is_null().all()
    assert df["intercept"].is_null().all()


def test_linear_fit_agg(inputs):
    from ivory.optimize import linear_fit

    x, y = inputs
    df = DataFrame({"x": x, "y": y}, schema={"x": Float64, "y": Float64})
    n = len(x)
    a = [1] * (n // 2) + [2] * (n - n // 2)
    df = df.with_columns(a=Series(a))
    df = df.group_by("a", maintain_order=True).agg(
        linear_fit("x", "y").struct.unnest(),
    )

    p1, _ = curve_fit(lambda x, a, b: a * x + b, x[: n // 2], y[: n // 2])
    p2, _ = curve_fit(lambda x, a, b: a * x + b, x[n // 2 :], y[n // 2 :])

    x = df["slope"].to_list()[0]
    np.testing.assert_allclose(x, p1[0], rtol=1e-5)
    x = df["intercept"].to_list()[0]
    np.testing.assert_allclose(x, p1[1], rtol=1e-5)

    x = df["slope"].to_list()[1]
    np.testing.assert_allclose(x, p2[0], rtol=1e-5)
    x = df["intercept"].to_list()[1]
    np.testing.assert_allclose(x, p2[1], rtol=1e-5)
