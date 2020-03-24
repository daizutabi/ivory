# import numpy as np
# import optuna

# from ivory.callbacks.pruning import Pruning
from ivory.core.base import Base


class Experiment(Base):
    __slots__ = []  # type:ignore

    def set_client(self, client):
        if client.tracker:
            self.set_tracker(client.tracker)
        if client.tuner:
            self.set_tuner(client.tuner)

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

    # def create_objective(self, params):
    #     self.objective.set_params(params)
    #     create_params = self.objective.create_params
    #     create_run = self.create_run
    #     has_pruner = self.tuner.pruner is not None
    #
    #     def objective(trial):
    #         params = create_params(trial)
    #         run = create_run(params)
    #         if run.tracking:
    #             run.tracking.param_names = list(trial.params.keys())
    #             trial.set_user_attr("run_id", run.id)
    #         if has_pruner:
    #             run.objects["pruning"] = Pruning(trial, run.monitor.metrics)
    #         run.start()
    #         score = run.monitor.best_score
    #         if np.isnan(score):
    #             message = "Best score is nan"
    #             raise optuna.exceptions.TrialPruned(message)
    #         return score
    #
    #     return objective
    #
    # def create_study(self):
    #     study = self.tuner.create_study(self.name, self.objective.mode)
    #     if self.id:
    #         study.set_user_attr("experiment_id", self.id)
    #     return study
