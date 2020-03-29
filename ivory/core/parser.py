import ast
import re

from ivory import utils


class Parser:
    def __init__(self, values=True):
        self.values = values

    def parse_args(self, args=None, **kwargs):
        if args is None:
            self.args = {}
        elif isinstance(args, (list, tuple)):
            self.args = dict(arg.split("=") for arg in args)
        else:
            self.args = args.copy()
        self.args.update(kwargs)
        self.options = parse_options(self.args)
        if not self.values:
            return self
        self.values = parse_values(self.args.values())
        if len(self.args) == 0 or all([len(x) == 1 for x in self.values]):
            self.mode = "single"
        elif len(self.args) == 1:
            self.mode = "scan"
        else:
            self.mode = "prod"
        return self

    def parse_params(self, params):
        self.names = parse_names(self.args.keys(), params)
        return self

    def parse(self, args, params, **kwargs):
        self.parse_args(args, **kwargs)
        self.parse_params(params)
        return self


def parse_names(names, params):
    fullnames = []
    for name in names:
        fullname = utils.get_fullnames(params, name)
        if not fullname:
            raise ValueError(f"Unknown parameter name: {name}")
        fullnames.append(fullname)
    return fullnames


def parse_values(values):
    list_values = []
    for value in values:
        if not isinstance(value, (list, str)):
            value = [value]
        else:
            match = re.match(r"(\d+)-(\d+)", value)
            if match:
                value = list(range(int(match.group(1)), int(match.group(2)) + 1))
            elif "," in value:
                value = [literal_eval(x) for x in value.split(",")]
            else:
                value = [literal_eval(value)]
        list_values.append(value)
    return list_values


def parse_options(args):
    options = {}
    for key, value in args.items():
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        options[key] = value
    return options


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
