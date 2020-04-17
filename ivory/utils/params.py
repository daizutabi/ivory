from typing import Any, Dict


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
        k = attrs[-1]
        if k not in x:
            x[k] = value
        elif isinstance(x[k], str) and x[k].startswith("$"):
            x[k] = value
        elif type(x[k]) is not type(value):
            raise ValueError(f"different type: {type(x[k])} != {type(value)}")
        else:
            if isinstance(value, dict):
                x[k].update(value)
            else:
                x[k] = value


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


def get_fullnames(params, name, prefix="", dict_allowed=False):
    """Returns a fullname found first.

    Examples:
        >>> params = {'a': 1, 'b': {'c': {'d': 2, 'e': [1, 2, 3]}}, "x": {'d': 2}}
        >>> list(get_fullnames(params, 'a'))
        ['a']
        >>> list(get_fullnames(params, 'c'))
        []
        >>> list(get_fullnames(params, 'd'))
        ['b.c.d', 'x.d']
        >>> list(get_fullnames(params, 'e'))
        ['b.c.e']
        >>> list(get_fullnames(params, 'b.c.d'))
        ['b.c.d']
        >>> list(get_fullnames(params, 'c.d'))
        ['b.c.d']
        >>> list(get_fullnames(params, 'e.2'))
        ['b.c.e.2']
    """
    if "." in name:
        name, _, suffix = name.partition(".")
        for fullname in get_fullnames(params, name, dict_allowed=True):
            yield ".".join([fullname, suffix])
    elif not isinstance(params, dict):
        return
    elif name in params:
        if dict_allowed or not isinstance(params[name], dict):
            yield prefix + name
    else:
        for key in params:
            prefix_ = prefix + key + "."
            yield from get_fullnames(params[key], name, prefix_, dict_allowed)


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
    fullnames = list(get_fullnames(params, name))
    if fullnames:
        return dot_get(params, fullnames[0])


def create_update(params, **kwargs):
    update = {}
    for name, value in kwargs.items():
        for fullname in get_fullnames(params, name):
            update[fullname] = value
    return update


def match(params, **query):
    """Returns if params match the query or not.

    Avaliable query type:
        tuple: (start, top) range including the stop value.
        list: [a1, a2, ..., an] parameters set.
        other: exact match.

    Examples:
        >>> params = {'a': 1, 'b': {'c': {'d': 2}}}
        >>> match(params, a=1)
        True
        >>> match(params, a=2)
        False
        >>> match(params, a=(0, 3))
        True
        >>> match(params, a=(3, 4))
        False
        >>> match(params, a=[5, 6])
        False
        >>> match(params, a=[0, 1])
        True
    """
    for name, cond in query.items():
        value = get_value(params, name)
        if value is None:
            return False
        elif isinstance(cond, tuple):
            if value < cond[0] or value > cond[1]:
                return False
        elif isinstance(cond, list):
            if value not in cond:
                return False
        elif value != cond:
            return False
    return True
