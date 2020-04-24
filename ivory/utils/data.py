import pandas as pd


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


def mean_argmax(output, target, columns=None):
    if columns is None:
        columns = ["pred", "true"]
    df = mean(output)
    pred = argmax(df)
    true = mean(target)
    return pd.DataFrame({columns[0]: pred, columns[1]: true})


# import scipy.special
#
# signal = scipy.special.softmax(df.to_numpy(), axis=1)
#
# df = pd.DataFrame(signal, index=df.index)
