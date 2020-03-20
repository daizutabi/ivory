import subprocess
import sys

import click

import ivory

click_option_params_path = click.option(
    "-p",
    "--params",
    type=click.Path(exists=True),
    default="params.yaml",
    help="Parameter file path.",
)


@click.group(invoke_without_command=True)
@click.pass_context
@click_option_params_path
def cli(ctx, params):
    if "." not in sys.path:
        sys.path.insert(0, ".")
    if ctx.invoked_subcommand is None:
        click.echo("I was invoked without subcommand")
        print(params)


@cli.command(help="Start a single run.")
@click_option_params_path
def run(params):
    ivory.run(params)


@cli.command()
@click_option_params_path
def ui(params):
    tracker = ivory.create_tracker(params)
    tracking_uri = tracker.tracking_uri
    subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])


@cli.command(help="Show the parameter file contents.")
@click_option_params_path
def yaml(params):
    with open(params) as file:
        params_yaml = file.read()
    print(params_yaml)


def main():
    cli()


if __name__ == "__main__":
    main()
