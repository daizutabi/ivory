import click
import yaml as yml

from ivory import utils
from ivory.core.environment import Environment


def load_params(params_path):
    with open(params_path) as file:
        params_yaml = file.read()
    return utils.to_float(yml.safe_load(params_yaml))


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
    env = Environment(params_path)
    print(env)
    experiment = env.create_experiment()
    experiment.start()
    print(experiment)
    run = experiment.create_run()
    print(run)


@cli.command()
@click_option_params_path
def ui(params_path):
    print(params_path)


@cli.command()
@click_option_params_path
def list(params_path):
    print(params_path)
    a = load_params(params_path)
    print(a)


@cli.command(help="Show the parameter file contents.")
@click_option_params_path
def params(params_path):
    print(params_path)
    a = load_params(params_path)
    print(a)


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
