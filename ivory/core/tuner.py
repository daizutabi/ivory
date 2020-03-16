from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna

from ivory.callbacks.pruning import Pruning
from ivory.core import instance


@dataclass
class Tuner:
    objective: Callable = field(repr=False)
    name: str = ""
    storage: Optional[str] = None
    sampler: Optional[str] = None
    pruner: Optional[str] = None
    monitor: str = field(default="", init=False)
    direction: str = field(default="minimize", init=False)
    load_if_exists: bool = False
    optimize_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.study = None

    def create_study(self, name: str, monitor):
        """Creates and returns a Optuna Study object."""
        self.name = name
        monitor = instance.instantiate(monitor)
        self.monitor = monitor.monitor
        if monitor.mode == "min":
            self.direction = "minimize"
        elif monitor.mode == "max":
            self.direction = "maximize"
        else:
            raise ValueError("monitor's mode must be 'min' or 'max'.")
        self.study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.name,
            direction=self.direction,
            load_if_exists=self.load_if_exists,
        )
        return self.study

    def create_objective(self, create_run):
        objective = self.objective
        monitor = self.monitor
        has_pruner = self.pruner is not None

        def _objective(trial):
            objective(trial)
            objects = {}
            if has_pruner:
                objects["pruning"] = Pruning(trial, monitor)
            name = f"trial#{trial.number}"
            run = create_run(trial.params, name=name, objects=objects)
            if "tracking" in run:
                run.tracking.param_names = list(trial.params.keys())
            run.start()
            if run.id:
                trial.set_user_attr("run_id", run.id)
            score = run.monitor.best_score
            if np.isnan(score):
                message = "Best score is nan"
                raise optuna.exceptions.TrialPruned(message)
            return score

        return _objective

    def optimize(self, create_run):
        objective = self.create_objective(create_run)
        self.study.optimize(objective, **self.optimize_kwargs)
