from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import optuna

from ivory.callbacks import Pruning
from ivory.core import instance


@dataclass
class Optuna:
    objective: Callable = field(repr=False)
    name: str = field(init=False)
    storage: Optional[str] = None
    sampler: Optional[str] = None
    pruner: Optional[str] = None
    monitor: str = field(default="", init=False)
    direction: str = field(default="minimize", init=False)
    load_if_exists: bool = False
    optimize_kwargs: Dict[str, Any] = field(default_factory=dict)

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
            if has_pruner:
                callbacks = [Pruning(trial, monitor)]
            else:
                callbacks = None
            name = f"trial#{trial.number}"
            run = create_run(trial.params, callbacks=callbacks, name=name)
            run.start()
            if run.run_id:
                trial.set_user_attr("run_id", run.run_id)
            return run.monitor.best_score

        return _objective

    def optimize(self, create_run):
        objective = self.create_objective(create_run)
        self.study.optimize(objective, **self.optimize_kwargs)