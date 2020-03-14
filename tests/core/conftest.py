import os

import pytest


@pytest.fixture
def params_single():
    return {"data": {"call": "numpy.array", "object": [1, 2]}}


@pytest.fixture
def params():
    return {
        "data": {"call": "numpy.array", "object": [1, 2]},
        "series": {"class": "pandas.Series", "data": "$"},
        "metrics": {"class": "ivory.callbacks.Metrics"},
        "monitor": {"class": "ivory.callbacks.Monitor"},
        "a, b": {"call": "numpy.array", "object": [3, 4]},
    }


yaml = """
data:
  call: numpy.array
  object: [1, 2]
data2:
  call: numpy.array
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
