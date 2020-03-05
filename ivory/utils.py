import functools
from typing import Any, Dict

import hydra
import numpy as np
from sklearn.model_selection import KFold


def fold_array(splitter, x, y=None, groups=None):
    fold = np.full(x.shape[0], -1, dtype=np.int8)
    for i, (_, test_index) in enumerate(splitter.split(x, y, groups)):
        fold[test_index] = i
    return fold


def kfold_split(x, n_splits=5):
    splitter = KFold(n_splits, random_state=0, shuffle=True)
    return fold_array(splitter, x)


def _parse_params(config, exclude, objects=None):
    params = {}
    for key in config:
        if key != exclude:
            value = config[key]
            if isinstance(value, str):
                if value == "$":
                    if objects is None:
                        raise ValueError("objects argument required.")
                    value = objects[key]
                elif value.startswith("$."):
                    if objects is None:
                        raise ValueError("objects argument required.")
                    value = value[2:]
                    if "." in value:
                        key_, _, rest = value.partition(".")
                        value = eval(f"objects[key_].{rest}")
                    else:
                        value = objects[value]
            params[key] = value
    return params


def instantiate(config, objects=None):
    if "class" in config:
        cls = hydra.utils.get_class(config["class"])
        params = _parse_params(config, "class", objects)
        return cls(**params)
    elif "call" in config:
        func = hydra.utils.get_method(config.call)
        params = _parse_params(config, "call", objects)
        return func(**params)
    elif "function" in config:
        func = hydra.utils.get_method(config.function)
        params = _parse_params(config, "function", objects)
        return functools.partial(func, **params)
    else:
        return config


class Config:
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

    def __getitem__(self, key):
        return getattr(self, key)


def parse(config):
    objects: Dict[str, Any] = {}
    for key in config:
        if hasattr(config[key], "keys"):
            objects[key] = instantiate(config[key], objects)
        else:
            objects[key] = config[key]
    return Config(objects)
