import ast
import os
import re
import subprocess

import ivory.core.tracker
from ivory import utils
from ivory.core.instance import create_base_instance_chain


def run(params, source_name=""):
    bases = ["environment", "experiment", "run"]
    run = create_base_instance_chain(bases, params, source_name)[-1]
    run.start()


def chain(params, args, source_name=""):
    if isinstance(params, str):
        source_name = os.path.abspath(params)
        params = utils.load_params(params)
    names, values = parse_args(params, args)
    print(names)
    print(values)


def parse_args(params, args):
    names = []
    values = []
    for arg in args:
        name, value = arg.split("=")
        fullname = utils.get_fullname(params, name)
        if fullname is None:
            raise ValueError(f"Unknown params name: {name}")
        names.append(fullname)
        match = re.match(r"(\d+)-(\d+)", value)
        if match:
            value = list(range(int(match.group(1)), int(match.group(2)) + 1))
        elif "," in value:
            value = [ast.literal_eval(x) for x in value.split(",")]
        else:
            raise ValueError(f"Unknown value pattern: {value}")
        values.append(value)
    return names, values


def ui(params):
    tracker = ivory.core.tracker.create_tracker(params)
    tracking_uri = tracker.tracking_uri
    subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
