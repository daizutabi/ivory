import pytest
import ivory

import os


yaml = """
tracking:
  class: ivory.callbacks.Tracking
experiment:
  class: ivory.core.Experiment
  run_class: ivory.core.Run
"""


@pytest.fixture()
def experiment(tmpdir):
    path = os.path.join(tmpdir, "params.yaml")
    with open(path, 'w') as f:
        f.write(yaml)
    return ivory.create_experiment(path)
