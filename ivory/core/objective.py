import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core import instance


class Objective:
    def __init__(self, **suggests):
        self.suggests = suggests
        for key, value in self.suggests.items():
            if isinstance(value, str):
                self.suggests[key] = instance.get_attr(value)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(suggests={list(self.suggests.keys())})"

    def __call__(self, suggest_name, create_run, has_pruning):
        suggest = self.suggests[suggest_name]

        def objective(trial: Trial):
            suggest(trial)
            run = create_run(trial.params)
            if run.tracking:
                run.tracking.set_tags(run.id, {"trial_number": trial.number})
                trial.set_user_attr("run_id", run.id)
            if has_pruning:
                run["pruning"] = Pruning(trial, run.monitor.metric)
            run.start("train")
            score = run.monitor.best_score
            if np.isnan(score):
                raise optuna.exceptions.TrialPruned("Best score is nan.")
            return score

        return objective
