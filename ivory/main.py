import os
import sys

import click

from ivory.core.client import Client

if "." not in sys.path:
    sys.path.insert(0, ".")


def normpath(params):
    if "." not in params:
        params = params + ".yaml"
    if not os.path.exists(params):
        click.secho(f"No sufh file: {params}", fg="red", bold=True)
        sys.exit()
    return params


@click.group()
def cli():
    pass


@cli.command(help="Start product runs.")
@click.argument("params")
@click.argument("args", nargs=-1)
@click.option("-m", "--message", default="")
def run(params, args, message):
    Client(normpath(params)).product(args, message)


@cli.command(help="Start chain runs.")
@click.argument("params")
@click.argument("args", nargs=-1)
@click.option("-m", "--message", default="")
def chain(params, args, message):
    Client(normpath(params)).chain(args, message)


@cli.command(help="List runs.")
@click.argument("params")
@click.argument("args", nargs=-1)
@click.option("-m", "--message", default="")
def list(params, args, message):
    runs = Client(normpath(params)).list(args, message)
    for run in runs:
        click.echo(run)


@cli.command()
@click.argument("params")
def ui(params):
    Client(normpath(params)).ui()


@cli.command(help="Show the parameter file contents.")
@click.argument("params")
def show(params):
    params = normpath(params)
    with open(params) as file:
        params_yaml = file.read()
    print(params_yaml)


def main():
    cli()


if __name__ == "__main__":
    main()
