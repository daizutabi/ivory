import ast
import re

from ivory import utils


class Parser:
    def parse_args(self, args=None, **kwargs):
        if not args:
            self.args = {}
        elif isinstance(args, (list, tuple)):
            self.args = dict(arg.split("=") for arg in args)
        elif isinstance(args, dict):
            self.args = args.copy()
        else:
            raise ValueError(f"Invalid arguments type: {type(args)}.")
        self.args.update(kwargs)
        self.values = parse_values(self.args.values())
        counts = [len(x) for x in self.values]
        counts_without_one = [x for x in counts if x != 1]
        if len(self.args) == 0 or all(x == 1 for x in counts):
            self.mode = "single"
        elif len(self.args) == 1 or len(counts_without_one) == 1:
            self.mode = "scan"
        else:
            self.mode = "prod"
        return self

    def parse_params(self, params):
        self.update, self.fullnames, self.options = parse_names(
            self.args, self.values, params
        )
        return self

    def parse(self, args, params, **kwargs):
        self.parse_args(args, **kwargs)
        self.parse_params(params)
        return self


def parse_names(args, values, params):
    update = {}
    names = {}
    options = {}
    for name, value in zip(args, values):
        fullnames = utils.get_fullnames(params, name)
        if not fullnames:
            options[name] = value[0]
        else:
            names[name] = fullnames
            update[fullnames] = value
    return update, names, options


def parse_values(values):
    list_values = []
    for value in values:
        if not isinstance(value, (list, str)):
            value = [value]
        elif isinstance(value, str):
            match = re.match(r"(.+)-(.+)", value)
            if match:
                try:
                    start = literal_eval(match.group(1))
                    stop = literal_eval(match.group(2))
                    if isinstance(start, int) and isinstance(stop, int):
                        value = range(start, stop + 1)
                    else:
                        value = (start, stop)
                except Exception:
                    value = [literal_eval(value)]
            elif "," in value:
                value = [literal_eval(x) for x in value.split(",")]
            else:
                value = [literal_eval(value)]
        list_values.append(value)
    return list_values


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
