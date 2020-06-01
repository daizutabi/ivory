import ast
import contextlib
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

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


def local_file_uri_to_path(uri):
    """
    Convert URI to local filesystem path.
    No-op if the uri does not have the expected scheme.
    """
    path = urllib.parse.urlparse(uri).path if uri.startswith("file:") else uri
    return urllib.request.url2pathname(path)


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


def normpath(name: str, directory: str = "") -> str:
    """Returns the absolute path with the extension."""
    path = os.path.abspath(os.path.join(directory, name))
    if os.path.exists(path + ".yaml"):
        return path + ".yaml"
    if os.path.exists(path + ".yml"):
        return path + ".yml"
    else:
        return path


def load_params(name: str, source_name: str = "") -> Tuple[Params, str]:
    if source_name:
        directory = os.path.dirname(source_name)
        source_name = normpath(name, directory)
    else:
        source_name = name
    with open(source_name, "r") as file:
        params_yaml = file.read()
    params = yaml.safe_load(params_yaml)
    params = literal_eval(params)
    update_include(params, source_name)
    params = inherit(params, source_name)
    return params, source_name


def update_include(params, source_name, include=None):
    if "include" in params:
        name = params.pop("include")
        include = load_params(name, source_name)[0]
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


def inherit(params, source_name):
    if "extends" in params:
        return _inherit(params, source_name)
    for key, value in params.items():
        if isinstance(value, dict):
            params[key] = inherit(value, source_name)
    return params


def _inherit(params, source_name):
    base = {}
    for key, value in params.items():
        if key == 'extends':
            break
        base[key] = value
    name = params.pop("extends")
    base.update(load_params(name, source_name)[0])
    for key, value in base.items():
        if key in params:
            if value is None:
                base[key] = params[key]
            elif isinstance(value, dict):
                for k in params[key]:
                    value[k] = params[key][k]
    for key, value in params.items():
        if key not in base:
            base[key] = value
    return base


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
