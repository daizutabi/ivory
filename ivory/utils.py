from importlib import import_module
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import KFold


def fold_array(splitter, x, y=None, groups=None) -> np.ndarray:
    fold = np.full(x.shape[0], -1, dtype=np.int8)
    for i, (_, test_index) in enumerate(splitter.split(x, y, groups)):
        fold[test_index] = i
    return fold


def kfold_split(x, n_splits=5) -> np.ndarray:
    splitter = KFold(n_splits, random_state=0, shuffle=True)
    return fold_array(splitter, x)


def get_attr(path: str) -> type:
    module_path, _, name = path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, name)


def _parse_params(config: Dict[str, Any], objects: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key in config:
        if key not in ["class", "def"]:
            value = config[key]
            if isinstance(value, str):
                if value == "$":
                    value = objects[key]
                elif value.startswith("$."):
                    value = value[2:]
                    if "." in value:
                        key_, _, rest = value.partition(".")
                        value = eval(f"objects[key_].{rest}")
                    else:
                        value = objects[value]
            params[key] = value
    return params


def _instantiate(config: Dict[str, Any], objects: Dict[str, Any]) -> Any:
    if "class" in config:
        cls = get_attr(config["class"])
        return cls(**_parse_params(config, objects))
    elif "def" in config:
        func = get_attr(config["def"])
        return func(**_parse_params(config, objects))
    else:
        return config


class Config:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    def __getattr__(self, key: str):
        return self._config[key]

    def __getitem__(self, key: str):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config


def parse(
    config: List[Dict[str, Any]], default: Dict[str, Any] = None, keys: List[str] = None
) -> Config:
    objects: Dict[str, Any] = {}
    for sub in config:
        for key in sub:
            if keys and key not in keys:
                continue
            if default and key in default:
                objects[key] = default[key]
            elif hasattr(sub[key], "keys"):
                objects[key] = _instantiate(sub[key], objects)
            else:
                objects[key] = sub[key]
    return Config(objects)


def instantiate(
    config: List[Dict[str, Any]], name: str, default: Dict[str, Any] = None
) -> Any:
    if default is None:
        default = {}
    for sub in config:
        for key in sub:
            if key == name:
                return _instantiate(sub[key], default)
