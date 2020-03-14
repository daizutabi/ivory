import importlib
import re
from functools import partial
from typing import Any, Dict, List, Tuple

Map = Dict[str, Any]


def get_attr(path: str) -> type:
    if "." not in path:
        path = f"__main__.{path}"
    module_path, _, name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def get_classes(params: Map):
    for key in params:
        if isinstance(params[key], dict) and "class" in params[key]:
            yield get_attr(params[key]["class"])


def parse_value(value, objects: Map, key: str = None) -> Any:
    """
    Examples:
       >>> objects = {"a": 0, "b": [1, 2, 3]}
       >>> parse_value("$", objects, "a")
       0
       >>> parse_value("$.b", objects, "a")
       [1, 2, 3]
       >>> parse_value("$.b.1", objects, "a")
       2
       >>> parse_value("$.b.pop()", objects, "a")
       3
       >>> objects
       {'a': 0, 'b': [1, 2]}
    """
    if isinstance(value, str):
        if value == "$":
            if key is None:
                raise ValueError("Key not given.")
            return objects[key]
        if value.startswith("$."):
            value = value[2:]
            m = re.match(r"([^.]*)\.(\d+)$", value)
            if m:
                return objects[m.group(1)][int(m.group(2))]
            if "." in value:
                key, _, rest = value.partition(".")
                return eval(f"objects[key].{rest}")
            else:
                return objects[value]
    return value


def parse_params(params: Map, objects: Map) -> Map:
    """
    Examples:
       >>> objects = {"a": 0, "b": [1, 2, 3]}
       >>> parse_params({"a": "$", "b": "$.a", "c": "$.b.1"}, objects)
       {'a': 0, 'b': 0, 'c': 2}
       >>> parse_params({"a": ["$.a", "$.b"], "b": {"a": "$"}}, objects)
       {'a': [0, [1, 2, 3]], 'b': {'a': 0}}
       >>> parse_params({"a__b": "$"}, objects)
       {'a': 0, 'b': [1, 2, 3]}
    """
    parsed = {}
    for key in params:
        if key in ["class", "call", "def"]:
            continue
        value = params[key]
        if isinstance(value, dict):
            parsed[key] = instantiate(value, objects)
        elif isinstance(value, list):
            parsed[key] = [parse_value(v, objects) for v in value]  # type:ignore
        elif "__" in key and value == "$":
            for key in key.split("__"):
                parsed[key] = parse_value(value, objects, key)
        else:
            parsed[key] = parse_value(value, objects, key)
    return parsed


def _instantiate(params: Map, objects: Map) -> Any:
    if "class" in params:
        cls = get_attr(params["class"])
        return cls(**parse_params(params, objects))
    elif "call" in params:
        func = get_attr(params["call"])
        return func(**parse_params(params, objects))
    elif "def" in params:
        func = get_attr(params["def"])
        params = parse_params(params, objects)
        if params:
            return partial(func, **params)
        else:
            return func
    else:
        return params


def _unpack(objects, keys, created):
    if len(keys) == 1:
        objects[keys[0]] = created
    else:
        for key, obj in zip(keys, created):
            objects[key] = obj


def instantiate(params: Map, default: Map = None) -> Any:
    if "class" in params or "call" in params or "def" in params:
        return _instantiate(params, default or {})

    objects: Map = {}
    for key in params:
        keys = key.split("__")
        if default:
            if keys[0] in default:
                for key in keys:
                    objects[key] = default[key]
                continue
        if isinstance(params[key], dict):
            created = _instantiate(params[key], objects)
            _unpack(objects, keys, created)
        else:
            objects[key] = parse_value(params[key], objects)
    return objects


def resolve_params(params: Map, names: List[str]) -> Tuple[List[str], List[str]]:
    """Resolve comma-separated parameter names.

    Args:
        params (dict): parameters dictionary
        names (list): list of parameter names

    Returns:
        tuple:
            (list of keys for params, resolved parameter names)

    Examples:
        >>> params = {"a__b": 0, "c": 1, "d__e": 2}
        >>> names = ["a", "d"]
        >>> resolve_params(params, names)
        (['a__b', 'd__e'], ['a', 'b', 'd', 'e'])
    """
    update: List[str] = []
    params_keys: List[str] = []
    for key in params:
        key_splitted = key.split("__")
        for name in names:
            if name in key_splitted:
                if key not in params_keys:
                    params_keys.append(key)
                update.extend(key_splitted)
    return params_keys, update
