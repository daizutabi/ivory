import copy
import itertools
import subprocess

import ivory.core.tracker
from ivory import utils
from ivory.core.instance import create_base_instance_chain


def run(params):
    bases = ["environment", "experiment", "run"]
    run = create_base_instance_chain(params, bases)[-1]
    run.start()


@utils.autoload
def chain(params, source_name, args):
    experiment, params, args = reduce_experiment(params, source_name, args)
    number = 1
    for name in args:
        for value in args[name]:
            params_chain = copy.deepcopy(params)
            utils.update_dict(params_chain, {name: value})
            params_chain["name"] = f"chain#{number}"
            run = experiment.create_run(params_chain)
            if run.tracking:
                run.tracking.param_names = list(args.keys())
            run.start()
            number += 1


@utils.autoload
def product(params, source_name, args):
    experiment, params, args = reduce_experiment(params, source_name, args)
    number = 1
    for value in itertools.product(*args.values()):
        params_prod = copy.deepcopy(params)
        update = {key: value for key, value in zip(args.keys(), value)}
        utils.update_dict(params_prod, update)
        params_prod["name"] = f"product#{number}"
        run = experiment.create_run(params_prod)
        if run.tracking:
            run.tracking.param_names = list(args.keys())
        run.start()
        number += 1


def reduce_experiment(params, source_name, args):
    bases = ["environment", "experiment"]
    experiment = create_base_instance_chain(params, bases, source_name=source_name)[-1]
    params = params["run"]
    args = utils.parse_args(params, args)
    return experiment, params, args


def ui(params):
    tracker = ivory.core.tracker.create_tracker(params)
    tracking_uri = tracker.tracking_uri
    subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
