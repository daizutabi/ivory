from importlib import import_module
from typing import Any, Dict, List

Map = Dict[str, Any]


def get_attr(path: str) -> type:
    module_path, _, name = path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, name)


def parse_params(config: Map, objects: Map) -> Map:
    params = {}
    for key in config:
        if key in ["class", "def"]:
            continue
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


def _instantiate(config: Map, objects: Map) -> Any:
    if "class" in config:
        cls = get_attr(config["class"])
        return cls(**parse_params(config, objects))
    elif "def" in config:
        func = get_attr(config["def"])
        return func(**parse_params(config, objects))
    else:
        return config


def instantiate(config: Map, names: List[str] = None, default: Map = None) -> Any:
    if "class" in config or "def" in config:
        return _instantiate(config, default or {})

    objects: Map = {}
    if names:
        assert all(name in config for name in names)
    for key in config:
        if names and key not in names:
            continue
        if default and key in default:
            objects[key] = default[key]
        elif isinstance(config[key], dict):
            objects[key] = _instantiate(config[key], objects)
        else:
            objects[key] = config[key]
    return objects
