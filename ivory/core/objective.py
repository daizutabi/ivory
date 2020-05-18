import gc
from typing import Any, Callable, Dict

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core import instance
from ivory.utils.range import Range


class Objective:
    def __init__(self, **suggests):
        self.suggests = suggests
        for key, value in self.suggests.items():
            if isinstance(value, str):
                self.suggests[key] = instance.get_attr(value)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(suggests={list(self.suggests.keys())})"

    def __call__(
        self, suggest_name: str, create_run: Callable, has_pruning: bool
    ) -> Callable:
        suggest = self.suggests[suggest_name]

        def objective(trial: Trial):
            suggest(trial)
            run = create_run(trial.params)
            if run.tracking:
                run.tracking.set_tags(run.id, {"trial_number": trial.number})
                trial.set_user_attr("run_id", run.id)
            if has_pruning:
                run.set(pruning=Pruning(trial, run.monitor.metric))
            run.start("train")
            score = run.monitor.best_score
            if np.isnan(score):
                raise optuna.exceptions.TrialPruned("Best score is nan.")
            del run
            gc.collect()
            return score

        return objective

    def create_suggest(self, params: Dict[str, Any]) -> str:
        """Creates a suggest function from a parameter dictionary."""
        suggests = {}
        for key, value in params.items():
            log = False
            if key.endswith(".log"):
                log = True
                key = key.rpartition(".")[0]
            if isinstance(value, Range):
                low, high, step = value.start, value.stop, value.step
                if log:
                    suggests[key] = ["float", dict(low=low, high=high, log=log)]
                elif value.is_integer:
                    suggests[key] = ["int", dict(low=low, high=high, step=step)]
                else:
                    if step == 1 and value.num == 0:
                        suggests[key] = ["float", dict(low=low, high=high)]
                    else:
                        if value.num:
                            step = (high - low) / value.num
                        args = dict(low=low, high=high, step=step)
                        suggests[key] = ["discrete_uniform", args]
            else:
                suggests[key] = ["categorical", dict(choices=value)]

        def suggest(trial: Trial, suggests=suggests):
            for key, value in suggests.items():
                suggest = getattr(trial, "suggest_" + value[0])
                suggest(key, **value[1])

        suggest_name = ".".join(suggests.keys())
        self.suggests[suggest_name] = suggest
        return suggest_name
