import numpy as np
import pandas as pd

import ivory.utils.data


def test_softmax():
    df = pd.DataFrame([[1, 2], [3, 4]], index=[3, 4])
    df = ivory.utils.data.softmax(df)
    assert np.allclose(df.sum(axis=1).to_numpy(), [1, 1])


def test_mean():
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=[3, 4, 4])
    df = ivory.utils.data.mean(df)
    assert list(df.index) == [3, 4]
    assert np.allclose(df.to_numpy(), [[1, 2], [4, 5]])

    s = pd.Series([1, 2, 3], index=[3, 4, 4])
    s = ivory.utils.data.mean(s)
    assert isinstance(s, pd.Series)
    assert list(s.index) == [3, 4]
    assert np.allclose(s.to_numpy(), [1, 2.5])


def test_argmax():
    df = pd.DataFrame([[1, 2], [3, 4]], index=[3, 4])
    s = ivory.utils.data.argmax(df)
    assert s.loc[3] == 1
    assert s.loc[4] == 1
