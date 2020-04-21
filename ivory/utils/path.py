import ast
import contextlib
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, Tuple

import yaml

Params = Dict[str, Any]


def to_uri(path: str) -> str:
    if urllib.parse.urlparse(path).scheme:
        return path
    if "~" in path:
        path = os.path.expanduser(path)
    url = os.path.abspath(path)
    if "\\" in url:
        url = urllib.request.pathname2url(path)
    return urllib.parse.urlunparse(("file", "", url, "", "", ""))


@contextlib.contextmanager
def chdir(source_name: str):
    curdir = os.getcwd()
    if source_name:
        basedir = os.path.dirname(source_name)
        os.chdir(basedir)
    try:
        yield
    finally:
        os.chdir(curdir)


def normpath(path: str, directory: str = ".") -> str:
    """Returns the absolute path with the extension."""
    path = os.path.abspath(os.path.join(directory, path))
    if os.path.exists(path + ".yaml"):
        return path + ".yaml"
    if os.path.exists(path + ".yml"):
        return path + ".yml"
    else:
        return path


def load_params(path: str, source_name: str = "") -> Tuple[Params, str]:
    if source_name:
        directory = os.path.dirname(source_name)
        source_name = normpath(path, directory)
    else:
        source_name = path

    with open(source_name, "r") as file:
        params_yaml = file.read()
    params = yaml.safe_load(params_yaml)
    params = literal_eval(params)
    update_include(params, source_name)
    return params, source_name


def update_include(params, source_name, include=None):
    if "include" in params:
        path = params.pop("include")
        include = load_params(path, source_name)[0]
    elif include is None:
        include = {}
    for key, value in params.items():
        if key in include:
            if value is None:
                params[key] = include[key]
            elif isinstance(value, dict):
                for k in include[key]:
                    if k not in value:
                        value[k] = include[key][k]
        if isinstance(value, dict):
            update_include(value, source_name, include)


def literal_eval(x):
    if isinstance(x, dict):
        return {key: literal_eval(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [literal_eval(value) for value in x]
    elif isinstance(x, str):
        try:
            v = ast.literal_eval(x)
        except Exception:
            return x
        if isinstance(v, int):
            return x
        if isinstance(v, float) and "e" not in x and "E" not in x:
            return x
        return v
    else:
        return x


def params_iter(source_name: str, pattern: str = "") -> Iterator[Tuple[Params, str]]:
    with chdir(source_name):
        for file in os.listdir():
            path, ext = os.path.splitext(file)
            if ext not in [".yaml", ".yml"]:
                continue
            if pattern and not re.match(pattern, path):
                continue
            params, source_name_ = load_params(path, source_name)
            if "experiment" in params and "name" in params["experiment"]:
                yield params, source_name_
