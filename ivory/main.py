import logging
import sys

import click
import logzero
from logzero import logger

import ivory
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
@click.argument("path")
@click.argument("args", nargs=-1)
@click.option("-r", "--repeat", default=1, help="Number of repeatation.")
@click.option("--notest", is_flag=True, help="Skip inference after training.")
@click.option("-m", "--message", default="", help="Message for tracking.")
def run(path, args, repeat, notest, message):
    client = ivory.create_client()
    experiment = client.create_experiment(path)
    for run in experiment.start(args, repeat=repeat, message=message):
        run.start("train")
        if not notest:
            run = experiment.load_run(run.id, "best")
            try:
                run.start("test")
            except TestDataNotFoundError:
                pass


# @cli.command(help="Optimize hyper parameters.")
# @click.argument("path", callback=normpath)
# @click.argument("name")
# @click.argument("args", nargs=-1)
# @click.option("-m", "--message", default="", help="Message for tracking.")
# def optimize(path, name, args, message):
#     client = create_client(path)
#     client.optimize(name, args, message=message)
#
#
# @cli.command(help="Search runs.")
# @click.argument("path", callback=normpath)
# @click.argument("args", nargs=-1)
# @click.option("-m", "--message", default="", help="Message for tracking.")
# def search(path, args, message):
#     pass
#


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
