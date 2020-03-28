import functools

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core.instance import get_attr
from ivory.core.parser import Parser


class Objective:
    def __init__(self, sampler=None, pruner=None, **suggest):
        self.sampler = sampler
        self.pruner = pruner
        self.suggest = suggest
        for key, value in self.suggest.items():
            if isinstance(value, str):
                self.suggest[key] = get_attr(value)
            elif isinstance(value, list):
                value = {key: value}
            if isinstance(value, dict):
                self.suggest[key] = create_suggest(key, value)

    def create_update(self, trial: Trial, name, params):
        self.suggest[name](trial)
        parser = Parser().parse(trial.params, params["run"], values=False)
        return dict(zip(parser.names, parser.options.values()))

    def create_objective(self, name, params, create_run):
        create_update = functools.partial(self.create_update, name=name, params=params)
        tags = {"suggest": name}

        def objective(trial: Trial):
            update = create_update(trial)
            run = create_run(update, "trial", trial.number, trial.params.keys(), tags)
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


def create_suggest(key, params):
    suggests = []
    for x, value in params.items():
        suggest = [f"suggest_{value[0]}", x]
        if value[0] == "categorical":
            suggest += [value[1:]]
        else:
            suggest += value[1:]
        suggests.append(suggest)

    def suggest(trial):
        for s in suggests:
            getattr(trial, s[0])(*s[1:])

    return suggest
