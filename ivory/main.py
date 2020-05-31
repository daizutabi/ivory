import logging
import sys

import click
import logzero
from logzero import logger

import ivory
from ivory.core import parser

if "." not in sys.path:
    sys.path.insert(0, ".")


def loglevel(ctx, param, value):
    if param.name == "quiet" and value is True:
        logzero.loglevel(logging.WARNING)
    elif param.name == "verbose" and value is True:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    return value


@click.group()
def cli():
    pass


@cli.command(help="Invoke a run or product runs.")
@click.argument("name")
@click.argument("args", nargs=-1)
@click.option("-r", "--repeat", default=1, help="Number of repeatation.")
@click.option("-n", "--number", default=None, help="Task number to load.")
@click.option("-q", "--quiet", is_flag=True, help="Queit mode.", callback=loglevel)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.", callback=loglevel)
def run(name, args, repeat, number, quiet, verbose):
    client = ivory.create_client()
    if not args and repeat == 1:
        run = client.create_run(name)
        run.start("both")
    else:
        task = client.create_task(name, number and int(number))
        params = parser.parse_args(args)
        for run in task.product(params, repeat=repeat):
            run.start("both")


@cli.command(help="Optimize hyper parameters.")
@click.argument("name")
@click.argument("args", nargs=-1)
@click.option("-n", "--number", default=None, help="Study number to load.")
@click.option("-q", "--quiet", is_flag=True, help="Queit mode.", callback=loglevel)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.", callback=loglevel)
def optimize(name, args, number, quiet, verbose):
    client = ivory.create_client()
    run_number = number and int(number)
    if "=" not in args[0]:
        study = client.create_study(name, run_number=run_number)
        suggest_name = args[0]
        params = parser.parse_args(args[1:])
        kwargs = {key: values[0] for key, values in params.items()}
        if "n_trials" not in kwargs and "timeout" not in kwargs:
            kwargs["n_trials"] = 3  # FIXME
        study.optimize(suggest_name, **kwargs)
    else:
        params = parser.parse_args(args)  # FIXME: exclude len(params)==1
        params = list(zip(*params.items()))
        params = {params[0]: params[1]}
        study = client.create_study(name, params, run_number=run_number)
        study.optimize(n_trials=3)  # FIXME


@cli.command(help="Kill the last run to prune.")
@click.argument("name")
def kill(name):
    client = ivory.create_client()
    client.set_terminated(name, status="KILLED", run=-1)


@cli.command(help="Update parameters.")
@click.argument("name")
def update(name):
    client = ivory.create_client()
    client.update_params(name)


@cli.command(help="Remove deleted runs.")
@click.argument("name")
def clean(name):
    client = ivory.create_client()
    num_runs = client.remove_deleted_runs(name)
    print("Removed runs:", num_runs)


@cli.command(help="Start tracking UI.")
@click.option("-q", "--quiet", is_flag=True, help="Queit mode.", callback=loglevel)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.", callback=loglevel)
def ui(quiet, verbose):
    logger.info("Tracking UI.")
    client = ivory.create_client()
    client.ui()


def main():
    cli()


if __name__ == "__main__":
    main()
