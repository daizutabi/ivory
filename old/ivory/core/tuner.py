from dataclasses import dataclass, field
from typing import Optional

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from ivory.core.run import Study

DEFAULT_STORAGE = "sqlite://"


@dataclass
class Tuner:
    storage: str = field(default_factory=lambda: DEFAULT_STORAGE)
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    load_if_exists: bool = True

    def create_study(self, study_name: str, mode: str) -> Study:
        """Creates and returns an Optuna `Study` object."""
        if mode == "min":
            direction = "minimize"
        elif mode == "max":
            direction = "maximize"
        else:
            raise ValueError("Mode must be 'min' or 'max'.")
        study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction=direction,
            load_if_exists=self.load_if_exists,
        )
        return study

    def delete_study(self, study_name: str):
        optuna.delete_study(storage=self.storage, study_name=study_name)
