import ast
import re

from ivory.utils.range import Range


def parse_args(args=None, **kwargs):
    if args is None:
        args = {}
    elif isinstance(args, (list, tuple)):
        args = dict(arg.split("=") for arg in args)
    elif isinstance(args, dict):
        args = args.copy()
    else:
        raise ValueError(f"Invalid arguments type: {type(args)}.")
    args.update(kwargs)
    return {arg: parse_value(value) for arg, value in args.items()}


def parse_value(value):
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str):
        return [value]
    if not value:
        return [""]
    if "," in value:
        values = []
        for v in value.split(","):
            values.extend(parse_value(v))
        return values
    if ":" in value:
        value, n = value.split(":")
        n = int(n)
    else:
        n = 0
    match = re.match(r"(.+)-(.+)", value)
    if match:
        if "-" in match.group(1):
            start, stop = match.group(1).split("-")
            step = match.group(2)
        else:
            start = match.group(1)
            stop = match.group(2)
            step = 1
        start = literal_eval(start)
        stop = literal_eval(stop)
        step = literal_eval(step)
        if all(isinstance(x, (int, float)) for x in [start, stop, step]):
            return Range(start, stop, step, n)
    return [literal_eval(value)]


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
