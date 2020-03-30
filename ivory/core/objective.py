import functools

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core.instance import get_attr
from ivory.core.parser import Parser


class Objective:
    def __init__(self, sampler=None, pruner=None, **suggests):
        self.sampler = sampler
        self.pruner = pruner
        self.suggests = suggests
        for key, value in self.suggests.items():
            if isinstance(value, str):
                self.suggests[key] = get_attr(value)

    def optimize(
        self, study_name, name, update, options, params, create_run, tuner, mode
    ):
        objective = self.create_objective(name, update, params, create_run)
        study = tuner.create_study(
            study_name, mode, sampler=self.sampler, pruner=self.pruner,
        )
        study.optimize(objective, **options)
        return study

    def create_objective(self, name, update, params, create_run):
        create_update = functools.partial(
            self.create_update, name=name, update=update, params=params
        )

        def objective(trial: Trial):
            update, args = create_update(trial)
            run = create_run(update, "trial", trial.number, args)
            if run.tracking:
                trial.set_user_attr("run_id", run.id)
            if self.pruner:
                run["pruning"] = Pruning(trial, run.monitor.metric)
            run.start(leave=False)
            score = run.monitor.best_score
            if np.isnan(score):
                message = "Best score is nan"
                raise optuna.exceptions.TrialPruned(message)
            return score

        return objective

    def create_update(self, trial: Trial, name, update, params):
        self.suggests[name](trial)
        parser = Parser().parse(trial.params, params["run"])
        update = update.copy()
        for fullnames, values in parser.update.items():
            for fullname in fullnames:
                update[fullname] = values[0]
        return update, parser.fullnames.keys()
