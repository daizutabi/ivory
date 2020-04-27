import ast
import itertools
import re


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
    if not isinstance(value, (list, str)):
        value = [value]
    elif isinstance(value, str):
        if "," in value:
            values = []
            for v in value.split(","):
                values.extend(parse_value(v))
            return values
        match = re.match(r"(.+)-(.+)", value)
        if match:
            try:
                start = literal_eval(match.group(1))
                stop = literal_eval(match.group(2))
                if isinstance(start, int) and isinstance(stop, int):
                    if stop >= start:
                        value = range(start, stop + 1)
                    else:
                        value = range(start, stop - 1, -1)
                else:
                    value = (start, stop)
            except Exception:
                value = [literal_eval(value)]
        else:
            value = [literal_eval(value)]
    return value


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value


def product(args=None, **kwargs):
    params = parse_args(args, **kwargs)
    for values in itertools.product(*params.values()):
        update = {}
        for name, value in zip(params.keys(), values):
            update[name] = value
        yield update
