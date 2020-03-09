__version__ = "0.1.2"
from typing import Optional

from ivory.core.experiment import Experiment, create_experiment
from ivory.core.run import Run, create_run

__all__ = ["create_experiment", "create_run"]

active_experiment: Optional[Experiment] = None
active_run: Optional[Run] = None
