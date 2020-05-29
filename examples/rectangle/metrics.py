import numpy as np


def mean_squared_error(true, pred):
    return np.mean((true - pred) ** 2)
