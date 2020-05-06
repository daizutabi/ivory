from click.testing import CliRunner

from ivory import main, utils


def test_run():
    runner = CliRunner()
    with utils.chdir("tests/a"):
        result = runner.invoke(main.cli, ["run", "example", "fold=1,2"])
        assert result.exit_code == 0


def test_optimize():
    runner = CliRunner()
    with utils.chdir("tests/a"):
        result = runner.invoke(main.cli, ["optimize", "example", "lr"])
        assert result.exit_code == 0
