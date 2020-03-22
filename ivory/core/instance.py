import importlib
import os
import re
from functools import partial
from typing import List

from ivory import utils
from ivory.core.base import Base
from ivory.core.default import update_class


def get_attr(path: str):
    if "." not in path:
        raise ValueError("module path not included")
    module_path, _, name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def instantiate(params, globals=None, kwargs=None):
    if globals is None:
        globals = {}
    else:
        globals = globals.copy()
    if kwargs is None:
        kwargs = {}
    return _instantiate(params, globals, kwargs)


def _instantiate(params, globals, kwargs):
    if "class" in params:
        key = "class"
    elif "call" in params:
        key = "call"
    elif "def" in params:
        key = "def"
    else:
        raise ValueError("dict-key must include one of (class, call, def)")

    attr = get_attr(params[key])
    args = {k: v for k, v in params.items() if k != key}
    args = parse_value(args, globals, "")
    if key != "def":
        return attr(**args, **kwargs)
    else:
        if args or kwargs:
            return partial(attr, **args, **kwargs)  # **args is correct.
        else:
            return attr


def parse_value(value, globals, key: str):
    """
    Examples:
       >>> globals = {"a": 0, "b": [1, 2, 3]}
       >>> parse_value("$", globals, "a")
       0
       >>> parse_value("$.b", globals, "a")
       [1, 2, 3]
       >>> parse_value("$.b.1", globals, "a")
       2
       >>> parse_value("$.b.pop()", globals, "a")
       3
       >>> globals
       {'a': 0, 'b': [1, 2]}
    """
    if isinstance(value, dict):
        if "class" in value or "call" in value or "def" in value:
            obj = globals[key] = _instantiate(value, globals, {})
            return obj
        else:
            return {key: parse_value(value[key], globals, key) for key in value}
    elif isinstance(value, list):
        return [parse_value(v, globals, key) for v in value]
    elif value == "$":
        return globals[key]
    elif isinstance(value, str) and value.startswith("$."):
        value = value[2:]
        m = re.match(r"(.*)\.(\d+)$", value)
        if m:
            value, index = m.group(1), int(m.group(2))
        else:
            index = -1
        if "." in value:
            key, _, rest = value.partition(".")
            value = eval(f"globals[key].{rest}")
        else:
            value = globals[value]
        if index >= 0:
            value = value[index]
    return value


def create_base_instance(base_name: str, params, source_name=""):
    if isinstance(params, str):
        source_name = os.path.abspath(params)
        params = utils.load_params(params)
    update_class(params)
    if base_name in params:
        params = params[base_name]
    kwargs = dict(params=params, source_name=source_name)
    return instantiate(params, kwargs=kwargs)


def create_base_instance_chain(base_names: list, params, source_name=""):
    if isinstance(params, str):
        source_name = os.path.abspath(params)
        params = utils.load_params(params)
    chain: List[Base] = []
    for base_name in base_names:
        if base_name not in params:
            continue
        if chain:
            instance = getattr(chain[-1], f"create_{base_name}")(params, source_name)
        else:
            instance = create_base_instance(base_name, params, source_name)
        chain.append(instance)
    return chain


def create_instance(name: str, params):
    if isinstance(params, str):
        params = utils.load_params(params)
    update_class(params)
    names = name.split(".")
    for name in names:
        params = params[name]
    return instantiate(params)
