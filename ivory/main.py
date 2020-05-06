import logging
import sys

import click
import logzero
from logzero import logger

import ivory
from ivory.core import parser
from ivory.core.exceptions import TestDataNotFoundError

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
@click.option("--notest", is_flag=True, help="Skip test after training.")
@click.option("--notrack", is_flag=True, help="No tracking mode.")
def run(name, args, repeat, notest, notrack):
    client = ivory.create_client(tracker=not notrack)
    task = client.create_task(name)
    params = parser.parse_args(args)
    for run in task.product(params, repeat=repeat):
        run.start("train")
        if not notest and not notrack:
            run = client.load_run(run.id, "best")
            try:
                run.start("test")
            except TestDataNotFoundError:
                pass


@cli.command(help="Optimize hyper parameters.")
@click.argument("path")
@click.argument("name")
@click.option("--notrack", is_flag=True, help="No tracking mode.")
def optimize(path, name, notrack):
    client = ivory.create_client(tracker=not notrack)
    experiment = client.create_experiment(path)
    study = experiment.create_study()
    study.optimize(name, n_trials=3)


@cli.command(help="Start tracking UI.")
@click.option("-q", "--quiet", is_flag=True, help="Queit mode.", callback=loglevel)
def ui(quiet):
    logger.info("Tracking UI.")
    client = ivory.create_client()
    client.ui()


def main():
    cli()


if __name__ == "__main__":
    main()
