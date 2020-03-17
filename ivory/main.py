import subprocess
import sys

import click

import ivory

click_option_params_path = click.option(
    "-p",
    "--params-path",
    type=click.Path(exists=True),
    default="params.yaml",
    help="Parameter file path.",
)


@click.group(invoke_without_command=True)
@click_option_params_path
def cli(params_path):
    if "." not in sys.path:
        sys.path.insert(0, ".")


@cli.command()
@click_option_params_path
def ui(params_path):
    environment = ivory.create_environment(params_path)
    tracking_uri = environment.tracker.tracking_uri
    subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])


@cli.command(help="Show the parameter file contents.")
@click_option_params_path
def yaml(params_path):
    with open(params_path) as file:
        params_yaml = file.read()
    print(params_yaml)


def main():
    cli()


if __name__ == "__main__":
    main()
