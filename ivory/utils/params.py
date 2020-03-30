import ast
import os
from typing import Any, Dict

import yaml


def load_params(path: str):
    """Loads a parameter YAML file."""
    with open(path, "r") as file:
        params_yaml = file.read()
    params = literal_eval(yaml.safe_load(params_yaml))
    if "include" in params:
        include_path = params.pop("include")
        directory = os.path.dirname(os.path.abspath(path))
        include_path = os.path.join(directory, include_path)
        include = load_params(include_path)
        include.update(params)
        params = include
    return params


def literal_eval(x):
    if isinstance(x, dict):
        return {key: literal_eval(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [literal_eval(value) for value in x]
    elif isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
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
    update = dot_to_list(update)  # for optuna
    for key, value in update.items():
        x = org
        attrs = key.split(".")
        for attr in attrs[:-1]:
            x = x[attr]
        if attrs[-1] not in x:
            x[attrs[-1]] = value
        elif type(x[attrs[-1]]) is not type(value):
            raise ValueError(f"different type: {type(x[attrs[-1]])} != {type(value)}")
        else:
            if isinstance(value, dict):
                x[attrs[-1]].update(value)
            else:
                x[attrs[-1]] = value


def dot_to_list(x: Dict[str, Any]) -> Dict[str, Any]:
    """Converts suffix integers into a list.

    Examples:
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


def dot_flatten(x: Dict[str, Any], flattened=None, prefix="") -> Dict[str, Any]:
    """Flatten dict in dot-format.

    Examples:
        >>> params = {"model": {"name": "abc", "x": {"a": 1, "b": 2}}}
        >>> dot_flatten(params)
        {'model.name': 'abc', 'model.x.a': 1, 'model.x.b': 2}
    """
    if flattened is None:
        flattened = {}
    for key, value in x.items():
        if isinstance(value, dict):
            dot_flatten(x[key], flattened, prefix + key + ".")
        else:
            flattened[prefix + key] = value
    return flattened


def dot_get(x: Dict[str, Any], key: str):
    """Dot style dictionay access.

    Examples:
        >>> x = {"a": 1, "b": {"x": "abc", "y": 2, "z": [0, 1, 2]}}
        >>> dot_get(x, "a")
        1
        >>> dot_get(x, "b.x")
        'abc'
        >>> dot_get(x, "b.z.1")
        1
    """
    keys = key.split(".")
    for key in keys[:-1]:
        if key not in x:
            return None
        x = x[key]
    key = keys[-1]
    if "0" <= key[0] <= "9":
        return x[int(key)]  # type:ignore
    else:
        return x[key]


def get_fullname(params, name, prefix="", dict_allowed=False):
    """Returns a fullname found first.

    Examples:
        >>> params = {'a': 1, 'b': {'c': {'d': 2, 'e': [1, 2, 3]}}}
        >>> get_fullname(params, 'a')
        'a'
        >>> get_fullname(params, 'c')
        >>> get_fullname(params, 'd')
        'b.c.d'
        >>> get_fullname(params, 'e')
        'b.c.e'
        >>> get_fullname(params, 'b.c.d')
        'b.c.d'
        >>> get_fullname(params, 'c.d')
        'b.c.d'
        >>> get_fullname(params, 'e.2')
        'b.c.e.2'
    """
    if "." in name:
        name, _, suffix = name.partition(".")
        fullname = get_fullname(params, name, dict_allowed=True)
        if fullname:
            return ".".join([fullname, suffix])
    elif not isinstance(params, dict):
        return
    elif name in params:
        if dict_allowed or not isinstance(params[name], dict):
            return prefix + name
    else:
        for key in params:
            prefix_ = prefix + key + "."
            fullname = get_fullname(params[key], name, prefix_, dict_allowed)
            if fullname:
                return fullname
        else:
            return


def get_fullnames(params, name):
    """Returns a tuple of fullnames.

    Examples:
        >>> params = {'a': 1, 'b': {'x': 2}, 'c': {'x': 2}}
        >>> get_fullnames(params, 'x')
        ('b.x', 'c.x')
    """
    fullnames = []
    for key in params:
        fullname = get_fullname({key: params[key]}, name)
        if fullname:
            fullnames.append(fullname)
    return tuple(fullnames)


def get_value(params, name):
    """
    Examples:
        >>> params = {'a': 1, 'b': {'c': {'d': 2}}}
        >>> get_value(params, 'a')
        1
        >>> get_value(params, 'd')
        2
        >>> get_value(params, 'b.c.d')
        2
        >>> get_value(params, 'c.d')
        2
    """
    fullname = get_fullname(params, name)
    return dot_get(params, fullname)
