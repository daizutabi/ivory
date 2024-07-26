import pandas as pd
import scipy.special


def softmax(df):
    prob = scipy.special.softmax(df.to_numpy(), axis=1)
    return pd.DataFrame(prob, index=df.index)


def mean(df):
    is_series = isinstance(df, pd.Series)
    df = df.reset_index().groupby("index").mean()
    df.sort_index(inplace=True)
    df.index.name = None
    if is_series:
        df = df[0]
    return df


def argmax(df):
    pred = df.to_numpy().argmax(axis=1)
    return pd.Series(pred, index=df.index)
