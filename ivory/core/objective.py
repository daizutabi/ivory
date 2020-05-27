import gc
from typing import Callable, Dict, Iterable

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.callbacks.pruning import Pruning
from ivory.core import instance
from ivory.utils.range import Range


class Objective:
    def __init__(self, **suggests):
        self.suggests = {}
        self.update(**suggests)

    def update(self, args=None, **suggests):
        if args is None:
            args = {}
        args.update(suggests)
        for key, value in args.items():
            if callable(value):
                self.suggests[key] = value
            elif isinstance(value, str):
                self.suggests[key] = instance.get_attr(value)
            elif isinstance(key, str):
                self.create_suggest({key: value})  # single parameter
            else:
                self.create_suggest(dict(zip(key, value)))  # product

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

    def create_suggest(self, params: Dict[str, Iterable]) -> str:
        """Creates a suggest function from a parameter dictionary."""
        suggests = {}
        for key, value in params.items():
            if isinstance(value, tuple):
                low, high = value
                type = "int" if isinstance(low, int) else "float"
                suggests[key] = [type, dict(low=low, high=high)]
            elif isinstance(value, range):
                low, high, step = value.start, value.stop - 1, value.step
                suggests[key] = ["int", dict(low=low, high=high, step=step)]
            elif isinstance(value, Range):
                low, high, step = value.start, value.stop, value.step
                if value.log:
                    suggests[key] = ["float", dict(low=low, high=high, log=True)]
                elif value.is_integer:
                    suggests[key] = ["int", dict(low=low, high=high, step=step)]
                else:
                    if step == 1 and value.num == 0:
                        suggests[key] = ["float", dict(low=low, high=high)]
                    else:
                        if value.num:
                            step = (high - low) / value.num  # type:ignore
                        args = dict(low=low, high=high, step=step)
                        suggests[key] = ["discrete_uniform", args]
            else:
                suggests[key] = ["categorical", dict(choices=list(value))]

        def suggest(trial: Trial, suggests=suggests):
            for key, value in suggests.items():
                suggest = getattr(trial, "suggest_" + value[0])
                suggest(key, **value[1])

        suggest_name = ".".join(suggests.keys())
        self.suggests[suggest_name] = suggest
        return suggest_name
