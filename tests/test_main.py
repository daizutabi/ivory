from click.testing import CliRunner

import ivory.utils.path
from ivory import main


def test_run():
    runner = CliRunner()
    with ivory.utils.path.chdir("tests/examples/a"):
        result = runner.invoke(main.cli, ["run", "example"])
    assert result.exit_code == 0


def test_task():
    runner = CliRunner()
    with ivory.utils.path.chdir("tests/examples/a"):
        result = runner.invoke(main.cli, ["run", "example", "fold=1,2"])
    assert result.exit_code == 0


def test_optimize():
    runner = CliRunner()
    with ivory.utils.path.chdir("tests/examples/a"):
        result = runner.invoke(main.cli, ["optimize", "example", "lr", "-v"])
    assert result.exit_code == 0

# def test_optimize_params():
#     runner = CliRunner()
#     with ivory.utils.path.chdir("tests/examples/a"):
#         args = ["optimize", "example", "lr.log=0.01-0.03", "fold=2", "-q"]
#         result = runner.invoke(main.cli, args)
#     assert result.exit_code == 0


# def test_optimize_params_for_pruning():
#     runner = CliRunner()
#     with ivory.utils.path.chdir("tests/examples/a"):
#         args = ["optimize", "example", "lr.log=0.01-0.03", "n_trials=15", "-q"]
#         result = runner.invoke(main.cli, args)
#     assert result.exit_code == 0


def test_clean(client):
    run = client.create_run("rfr")
    run.start()
    name = run.name
    client.tracker.client.delete_run(run.id)
    runner = CliRunner()
    with ivory.utils.path.chdir("tests/examples/a"):
        args = ["clean", "rfr"]
        result = runner.invoke(main.cli, args)
    assert result.exit_code == 0
    run = client.create_run("rfr")
    assert run.name == name
