from importlib import import_module
from typing import Any, Dict, List

Map = Dict[str, Any]


def get_attr(path: str) -> type:
    module_path, _, name = path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, name)


def get_classes(params: Map):
    for key in params:
        if isinstance(params[key], dict) and "class" in params[key]:
            yield get_attr(params[key]["class"])


def parse_params(params: Map, objects: Map) -> Map:
    parsed = {}
    for key in params:
        if key in ["class", "def"]:
            continue
        value = params[key]
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
        parsed[key] = value
    return parsed


def _instantiate(params: Map, objects: Map) -> Any:
    if "class" in params:
        cls = get_attr(params["class"])
        return cls(**parse_params(params, objects))
    elif "def" in params:
        func = get_attr(params["def"])
        return func(**parse_params(params, objects))
    else:
        return params


def instantiate(params: Map, names: List[str] = None, default: Map = None) -> Any:
    if "class" in params or "def" in params:
        return _instantiate(params, default or {})

    objects: Map = {}
    if names:
        assert all(name in params for name in names)
    for key in params:
        if names and key not in names:
            continue
        if default and key in default:
            objects[key] = default[key]
        elif isinstance(params[key], dict):
            objects[key] = _instantiate(params[key], objects)
        else:
            objects[key] = params[key]
    return objects
