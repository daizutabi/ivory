import nnabla.functions as F


def mse(input, target):
    return F.mean(F.squared_error(input, target))
