import re
from typing import Any, Dict

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


def to_float(x):
    if isinstance(x, dict):
        return {key: to_float(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [to_float(value) for value in x]
    elif isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    else:
        return x


def update_dict(org: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Update dict using dot-notation.

    Examples:
        >>> x = {"a": 1, "b": {"x": "abc", "y": 2, "z": [0, 1, 2]}}
        >>> update_dict(x, {"b": {"z": [0]}, "b.x": "def"})
        >>> x
        {'a': 1, 'b': {'x': 'def', 'y': 2, 'z': [0]}}
    """
    for key, value in update.items():
        x = org
        attrs = key.split(".")
        for attr in attrs[:-1]:
            x = x[attr]
        if attrs[-1] not in x:
            pass
        elif type(x[attrs[-1]]) is not type(value):
            raise ValueError(f"different type: {type(x[attrs[-1]])} != {type(value)}")
        else:
            if isinstance(value, dict):
                x[attrs[-1]].update(value)
            else:
                x[attrs[-1]] = value


def dot_to_list(x: Dict[str, Any]) -> Dict[str, Any]:
    """Convert suffix integers into list.

    Example:
        >>> x = {"a.0": 1, "a.1": 3, "b.x.0": 10, "b.x.1": 20}
        >>> dot_to_list(x)
        {'a': [1, 3], 'b.x': [10, 20]}
    """
    update: Dict[str, Any] = {}
    for key, value in x.items():
        head, _, tail = key.rpartition(".")
        if "0" <= tail <= "9":
            index = int(tail)
            if index == 0:
                if head in update:
                    raise KeyError(key)
                update[head] = [value]
            elif head not in update or len(update[head]) != index:
                raise KeyError(key)
            else:
                update[head].append(value)
        else:
            update[key] = value
    return update


def format_name_by_dict(name: str, params: Dict[str, Any]):
    """Format name with `{xxx.yyy}` by dict.

    Examples:
       >>> name = r"{model.name}-{data.num_samples}"
       >>> params = {"model": {"name": "abc"}, "data": {"num_samples": 100}}
       >>> format_name_by_dict(name, params)
       'abc-100'
    """

    def replace(match):
        x = params
        for m in match.group(1).split("."):
            x = x[m]
        return str(x)

    return re.sub(r"\{(.*?)\}", replace, name)
