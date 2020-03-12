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
        "a, b": {"def": "numpy.array", "object": [3, 4]},
    }


yaml = """
data:
  def: numpy.array
  object: [1, 2]
data2:
  def: numpy.array
  object: [3, 4]
experiment:
  class: ivory.core.Experiment
  run_class: ivory.torch.Run
  shared: [data]
"""


@pytest.fixture()
def path(tmpdir):
    path = os.path.join(tmpdir, "params.yaml")
    with open(path, "w") as f:
        f.write(yaml)
    return path
