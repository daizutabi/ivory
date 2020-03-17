from dataclasses import dataclass
from typing import Optional

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler


@dataclass
class Tuner:
    storage: str
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    load_if_exists: bool = False

    def create_study(self, study_name: str, mode: str):
        """Creates and returns a Optuna Study object."""
        if mode == "min":
            direction = "minimize"
        elif mode == "max":
            direction = "maximize"
        else:
            raise ValueError("monitor's mode must be 'min' or 'max'.")
        return optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction=direction,
            load_if_exists=self.load_if_exists,
        )
