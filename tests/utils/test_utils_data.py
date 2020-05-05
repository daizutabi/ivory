import numpy as np
import pandas as pd

import ivory.utils.data


def test_softmax():
    df = pd.DataFrame([[1, 2], [3, 4]], index=[3, 4])
    df = ivory.utils.data.softmax(df)
    assert np.allclose(df.sum(axis=1).to_numpy(), [1, 1])


def test_argmax():
    df = pd.DataFrame([[1, 2], [3, 4]], index=[3, 4])
    s = ivory.utils.data.argmax(df)
    print(s)
    assert s.loc[3] == 1
    assert s.loc[4] == 1
