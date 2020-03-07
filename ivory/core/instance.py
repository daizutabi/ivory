from importlib import import_module
from typing import Any, Dict, List

from ivory.core.config import Config

Map = Dict[str, Any]


def get_attr(path: str) -> type:
    module_path, _, name = path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, name)


def parse_params(config: Map, objects: Map) -> Map:
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


def _instantiate(config: Map, objects: Map) -> Any:
    if "class" in config:
        cls = get_attr(config["class"])
        return cls(**parse_params(config, objects))
    elif "def" in config:
        func = get_attr(config["def"])
        return func(**parse_params(config, objects))
    else:
        return config


def instantiate(config: Map, keys: List[str] = None, default: Map = None) -> Config:
    objects: Map = {}
    for key in config:
        if keys and key not in keys:
            continue
        if default and key in default:
            objects[key] = default[key]
        elif isinstance(config[key], dict):
            objects[key] = _instantiate(config[key], objects)
        else:
            objects[key] = config[key]
    return Config(objects)
