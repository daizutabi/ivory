import pandas as pd
import scipy.special


def concat_results(iterable):
    outputs = []
    targets = []
    for results in iterable:
        output, target = results.to_dataframe()
        outputs.append(output)
        targets.append(target)
    output = pd.concat(outputs)
    target = pd.concat(targets)
    output.sort_index(inplace=True)
    target.sort_index(inplace=True)
    return output, target


def softmax(df):
    prob = scipy.special.softmax(df.to_numpy(), axis=1)
    return pd.DataFrame(prob, index=df.index)


def mean(df):
    is_series = isinstance(df, pd.Series)
    df = df.reset_index().groupby("index").mean()
    df.index.name = None
    if is_series:
        df = df[0]
    return df


def argmax(df):
    pred = df.to_numpy().argmax(axis=1)
    return pd.Series(pred, index=df.index)


def mean_argmax(output, target=None, columns=None):
    df = mean(output)
    pred = argmax(df)
    if target is None:
        return pred
    if columns is None:
        columns = ["pred", "true"]
    true = mean(target)
    return pd.DataFrame({columns[0]: pred, columns[1]: true})
