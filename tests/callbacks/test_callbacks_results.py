import numpy as np
import pandas as pd

from ivory.callbacks.results import Results


def test_to_dataframe():
    n = 10
    index = np.arange(n)
    output = np.arange(5 * n).reshape(n, -1)
    target = np.arange(n)
    results = Results()
    results["val"] = dict(index=index, output=output, target=target)
    results["test"] = dict(index=index + n, output=output, target=target)

    output, target = results.to_dataframe()
    assert output.shape == (20, 5)
    assert target.shape == (20,)
    assert isinstance(output, pd.DataFrame)
    assert isinstance(target, pd.Series)

    results["test"] = dict(index=index + n, output=output, target=None)
    output, target = results.to_dataframe()
    assert output.shape == (20, 5)
    assert target.shape == (10,)
    target = np.arange(3 * n).reshape(n, -1)
    results["val"] = dict(index=index, output=output, target=target)
    results["test"] = dict(index=index + n, output=output, target=target)
    output, target = results.to_dataframe()
    assert isinstance(target, pd.DataFrame)

    index = np.arange(3 * n).reshape(n, -1)
    output = np.arange(15 * n).reshape(n, -1, 3)
    target = np.arange(3 * n).reshape(n, -1)
    results["val"] = dict(index=index, output=output, target=target)
    results["test"] = dict(index=index + n, output=output, target=target)
    output, target = results.to_dataframe()
    assert output.shape == (n * 3 * 2, 5)
    assert target.shape == (60,)
