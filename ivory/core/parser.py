import ast
import re

from ivory import utils


class Parser:
    def parse_args(self, args, values=True):
        if isinstance(args, list):
            self.args = dict(arg.split("=") for arg in args)
        else:
            self.args = args
        self.options = parse_options(self.args)
        if not values:
            return self
        self.values = parse_values(self.args.values())
        if len(args) == 0 or all([len(x) == 1 for x in self.values]):
            self.mode = "single"
        elif len(args) == 1:
            self.mode = "scan"
        else:
            self.mode = "product"
        return self

    def parse_params(self, params):
        self.names = parse_names(self.args.keys(), params)
        return self

    def parse(self, args, params, values=True):
        self.parse_args(args, values)
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
        match = re.match(r"(\d+)-(\d+)", value)
        if match:
            value = list(range(int(match.group(1)), int(match.group(2)) + 1))
        elif "," in value:
            value = [ast.literal_eval(x) for x in value.split(",")]
        else:
            try:
                value = [ast.literal_eval(value)]
            except ValueError:
                value = [value]
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
