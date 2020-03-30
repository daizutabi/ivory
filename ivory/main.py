# import datetime
import logging
import os
import sys

import click
import logzero
from logzero import logger

from ivory.core.client import create_client

if "." not in sys.path:
    sys.path.insert(0, ".")


def normpath(ctx, param, path):
    if not path:
        path = "client"
    path += ".yaml"
    if not os.path.exists(path):
        raise click.BadParameter(f"File not found: {path}")
    return path


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
@click.argument("path", callback=normpath)
@click.argument("args", nargs=-1)
@click.option("-r", "--repeat", default=1, help="Number of repeatation.")
@click.option("-t", "--test", is_flag=True, help="Infer after training.")
@click.option("-m", "--message", default="", help="Message for tracking.")
def run(path, args, repeat, test, message):
    client = create_client(path)
    for run in client.run(args, repeat=repeat, message=message):
        run.start(leave=False)
        if test:
            run = client.load_run(run.id, "best")
            run.start("test")


@cli.command(help="Optimize hyper parameters.")
@click.argument("path", callback=normpath)
@click.argument("name")
@click.argument("args", nargs=-1)
@click.option("-m", "--message", default="", help="Message for tracking.")
def optimize(path, name, args, message):
    client = create_client(path)
    client.optimize(name, args, message=message)


@cli.command(help="Search runs.")
@click.argument("path", callback=normpath)
@click.argument("args", nargs=-1)
@click.option("-m", "--message", default="", help="Message for tracking.")
def search(path, args, message):
    pass


@cli.command(help="Start tracking UI.")
@click.argument("path", callback=normpath)
@click.option("-q", "--quiet", is_flag=True, help="Queit mode.", callback=loglevel)
def ui(path, quiet):
    logger.info("Tracking UI.")
    client = create_client(path)
    client.ui()


# elif cmd in ["search", "list"]:         if "=" in args[0]:             mode = None
# else:             mode, args = args[0], args[1:]         parser =
# Parser().parse_args(args)         params = dict(zip(parser.args.keys(),
# parser.values))         for run in client.search_runs(params, mode, message,
# return_id=False):             run_id = run.info.run_id             start_dt =
# datetime.datetime.fromtimestamp(run.info.start_time / 1e3)             start_dt =
# start_dt.strftime("%Y-%m-%d %H:%M:%S")             print(run_id, start_dt)     elif
# cmd == "ui":         client.ui()


def main():
    cli()


if __name__ == "__main__":
    main()
