import ast
from typing import Any, Dict, Iterable, List

from ivory.utils.range import Range


def parse_args(args=None, **kwargs) -> Dict[str, Iterable[Any]]:
    if args is None:
        args = {}
    elif isinstance(args, (list, tuple)):
        args = dict(arg.split("=") for arg in args)
    elif isinstance(args, dict):
        args = args.copy()
    else:
        raise ValueError(f"Invalid arguments type: {type(args)}.")
    args.update(kwargs)
    parsed = {}
    for arg, value in args.items():
        value = parse_value(value)
        if arg.endswith(".log"):
            arg = arg[:-4]
            if not isinstance(value, Range):
                raise ValueError(f"Invalid value for log mode: {value}")
            value.log = True
        parsed[arg] = value
    return parsed


def parse_value(value) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return [value]
    if not value:
        return [""]
    if "," in value:
        values: List[Any] = []
        for v in value.split(","):
            values.extend(parse_value(v))
        return values
    try:
        return Range(value)
    except ValueError:
        return [literal_eval(value)]


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
