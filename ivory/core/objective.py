import functools

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core.instance import get_attr
from ivory.core.parser import Parser


class Objective:
    def __init__(self, **suggest):
        self.suggest = suggest
        for key, value in self.suggest.items():
            if isinstance(value, str):
                self.suggest[key] = get_attr(value)

    def create_update(self, trial: Trial, name, params):
        self.suggest[name](trial)
        parser = Parser().parse(trial.params, params["run"], values=False)
        return dict(zip(parser.names, parser.options.values()))

    def create_objective(self, name, params, create_run, has_pruner):
        create_update = functools.partial(self.create_update, name=name, params=params)
        tags = {"suggest": name}

        def objective(trial: Trial):
            update = create_update(trial)
            run = create_run(update, "trial", trial.number, trial.params.keys(), tags)
            if run.tracking:
                trial.set_user_attr("run_id", run.id)
            if has_pruner:
                run.set(pruning=Pruning(trial, run.monitor.metrics))
            run.start()
            score = run.monitor.best_score
            if np.isnan(score):
                message = "Best score is nan"
                raise optuna.exceptions.TrialPruned(message)
            return score

        return objective
