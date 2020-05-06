from click.testing import CliRunner

import ivory.utils.path
from ivory import main


# def test_run():
#     runner = CliRunner()
#     with ivory.utils.path.chdir("tests/a"):
#         result = runner.invoke(main.cli, ["run", "example", "fold=1,2"])
#         assert result.exit_code == 0
#
#
# def test_optimize():
#     runner = CliRunner()
#     with ivory.utils.path.chdir("tests/a"):
#         result = runner.invoke(main.cli, ["optimize", "example", "lr"])
#         assert result.exit_code == 0
