import os

import pytest


@pytest.fixture
def params_single():
    return {"data": {"def": "numpy.array", "object": [1, 2]}}


@pytest.fixture
def params():
    return {
        "data": {"def": "numpy.array", "object": [1, 2]},
        "series": {"class": "pandas.Series", "data": "$"},
        "metrics": {"class": "ivory.callbacks.Metrics", 'criterion': None},
    }


yaml = """
data:
  def: numpy.array
  object: [1, 2]
experiment:
  class: ivory.core.Experiment
  name: "example"
  run_class: ivory.core.Run
  run_name: "abc"
"""


@pytest.fixture()
def path(tmpdir):
    path = os.path.join(tmpdir, "params.yaml")
    with open(path, "w") as f:
        f.write(yaml)
    return path
