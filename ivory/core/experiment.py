import numpy as np
import optuna

from ivory.callbacks.pruning import Pruning
from ivory.core import instance
from ivory.core.base import Base
from ivory.core.run import create_run


class Experiment(Base):
    __slots__ = []  # type:ignore

    def set_tracker(self, tracker):
        self.objects["tracker"] = tracker
        if not self.name:
            self.name = "Default"
            self.params["name"] = self.name
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            self.params["id"] = self.id

    def set_tuner(self, tuner):
        self.objects["tuner"] = tuner

    def create_run(self, params="params.yaml"):
        run = create_run(params)
        if self.data:
            run.dataloader(self.data)
        if self.tracker:
            run.set_tracking(self.tracker, self.id)
        return run

    def create_objective(self, params="params.yaml"):
        self.objective.set_params(params)
        create_params = self.objective.create_params
        create_run = self.create_run
        has_pruner = self.tuner.pruner is not None

        def objective(trial):
            params = create_params(trial)
            run = create_run(params)
            if run.tracking:
                run.tracking.param_names = list(trial.params.keys())
                trial.set_user_attr("run_id", run.id)
            if has_pruner:
                run.objects["pruning"] = Pruning(trial, run.monitor.metrics)
            run.start()
            score = run.monitor.best_score
            if np.isnan(score):
                message = "Best score is nan"
                raise optuna.exceptions.TrialPruned(message)
            return score

        return objective

    def create_study(self):
        study = self.tuner.create_study(self.name, self.objective.mode)
        if self.id:
            study.set_user_attr("experiment_id", self.id)
        return study


create_experiment = instance.create_instance_factory("experiment")
