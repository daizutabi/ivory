import sys

import click

from ivory.core import client

if "." not in sys.path:
    sys.path.insert(0, ".")


def normpath(params):
    if "." not in params:
        return params + ".yaml"
    return params


@click.group()
def cli():
    pass


@cli.command(help="Start product runs.")
@click.argument("params")
@click.argument("args", nargs=-1)
def run(params, args):
    if args:
        client.product(normpath(params), args)
    else:
        client.run(normpath(params))


@cli.command(help="Start chain runs.")
@click.argument("params")
@click.argument("args", nargs=-1)
def chain(params, args):
    client.chain(normpath(params), args)


@cli.command()
@click.argument("params")
def ui(params):
    client.ui(normpath(params))


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
