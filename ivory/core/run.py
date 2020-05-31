"""
This module provides the `Run` class which is one of the main classes of
Ivory library. In addition, `Task` and `Study` classes are defined, which manages
multiple runs for cross validation, hyperparameter tuning, and so on.

To create an `Run` instance:

    import ivory

    client = ivory.create_run('example')

The argument `example` is an experiment name in which the created run is included.
Ivory assumes that `example.yml` or `example.yaml` file exists under the client's
working directory.

You can comfirm the client's working directory by:

    os.path.dirname(client.source_name)

One you got a `Run` instance. call `Run.start()` method to start training. For test,
call `Run.start('test')` instead. Also, you can perform traing and test by one step
with `Run.start('both')`.
"""
import functools
import gc
import inspect
import os
import warnings
from typing import Any, Dict, Iterable, Iterator, Optional

from termcolor import colored

import ivory.core.collections
import ivory.core.state
from ivory import utils
from ivory.core.base import CallbackCaller
from ivory.utils.tqdm import tqdm


class Run(CallbackCaller):
    """Run class, which is one of the main classes of Ivory library."""

    def set_tracker(self, tracker, name: str):
        """Sets tracker for tracking.

        By setting a tracker, a `Run` instance can be a run of MLFlow Tracking
        at the same time. MLFlow Tracking gives the Run ID and name for this
        instance.

        Args:
            tracker (Tracker): Tracker instance.
        """
        if not self.id:
            self.name, self.id = tracker.create_run(
                self.experiment_id, name, self.source_name
            )
            self.params[name]["name"] = self.name
            self.params[name]["id"] = self.id
        self.set(tracker=tracker)
        self.set(tracking=tracker.create_tracking())

    def init(self, mode: str = "train"):
        self.create_callbacks()
        self.mode = mode
        self.on_init_begin()
        self.on_init_end()

    def start(self, mode: str = "train"):
        """Starts traing and/or test.

        Args:
            mode: Mode name: `'train'`, `'test'`, or `'both'`.
        """
        if mode == "both":
            self.start("train")
            if self.tracker:
                self.tracker.load_state_dict(self, "best")
                self.start("test")
        else:
            self.init(mode)
            for obj in self.values():
                if hasattr(obj, "start") and callable(obj.start):
                    obj.start(self)

    def state_dict(self) -> dict:
        """Returns a state dictionary for all of member instances."""
        state_dict = {}
        for name, obj in self.items():
            if hasattr(obj, "state_dict") and callable(obj.state_dict):
                with warnings.catch_warnings():  # for torch LambdaLR scheduler
                    warnings.simplefilter("ignore")
                    state_dict[name] = obj.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads a state dictionary to all of member instances.

        Args:
            state_dict (dict): state dictionary for all of member instances.
        """
        for name in state_dict:
            with warnings.catch_warnings():  # for torch LambdaLR scheduler
                warnings.simplefilter("ignore")
                self[name].load_state_dict(state_dict[name])

    def save(self, directory: str):
        """Saves member instances.

        Args:
            directory: Directory where member instances are saved.
        """
        for name, state_dict in self.state_dict().items():
            path = os.path.join(directory, name)
            if hasattr(self[name], "save") and callable(self[name].save):
                self[name].save(state_dict, path)
            elif isinstance(self[name], ivory.core.state.State):
                ivory.core.state.save(state_dict, path)
            else:
                self.save_instance(state_dict, path)

    def save_instance(self, state_dict: Dict[str, Any], path: str):
        raise NotImplementedError

    def load(self, directory: str) -> Dict[str, Any]:
        """Loads member instances.

        Args:
            directory: Directory where member instances have been saved.
        """
        state_dict = {}
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if hasattr(self[name], "load") and callable(self[name].load):
                state_dict[name] = self[name].load(path)
            elif isinstance(self[name], ivory.core.state.State):
                state_dict[name] = ivory.core.state.load(path)
            else:
                instance_state_dict = self.load_instance(path)
                if instance_state_dict:
                    state_dict[name] = instance_state_dict
        return state_dict

    def load_instance(self, path):
        raise NotImplementedError


class Task(Run):
    """Task class creates a parent run that generates multiple runs."""

    def create_run(self, args, **kwargs) -> Run:  # type:ignore
        """Create a nested run according to arguments

        Args:
            args (dict, optional): Update dictionary.
            **kwargs: Additional update dictionary.

        Returns:
            Run: Created nested `Run` instance.
        """
        run = super().create_run(args, **kwargs)
        run_name = colored(f"[{run.name}]", "green")
        msg = utils.params.to_str(args)
        tqdm.write(run_name + f" {msg}")
        if self.tracking:
            self.tracking.set_parent_run_id(run.id, self.id)
        return run

    def terminate(self):
        if self.tracking:
            self.tracking.client.set_terminated(self.id)

    def product(
        self,
        params: Optional[Dict[str, Iterable[Any]]] = None,
        repeat: int = 1,
        **kwargs,
    ) -> Iterator[Run]:
        """Makes a product iterator.

        This iterator returns runs from cartesian product of input parameters.

        Args:
            params (dict, optional): Parameter range. Key is a parameter name and
                value is an iterable of parameter's value.
            repeat: Number of repeatation.
            **kwargs: Additional parameter range.

        See Also:
            [Product section](/tutorial/task#product) in Multiple Runs Tutorial
        """
        params, base_params = utils.params.split_params(params, **kwargs)
        if self.tracking:
            self.tracking.set_tags(self.id, params)
            self.tracking.set_tags(self.id, base_params)
        params_list = list(utils.params.product(params)) * repeat
        if "verbose" not in base_params or base_params["verbose"]:
            params_list = tqdm(params_list, desc="Prod ")
        for args_ in params_list:
            args = base_params.copy()
            args.update(args_)
            run = self.create_run(args)
            yield run
            del run
            gc.collect()
        self.terminate()

    def chain(
        self,
        params: Optional[Dict[str, Iterable[Any]]] = None,
        use_best_param: bool = True,
        **kwargs,
    ) -> Iterator[Run]:
        """Makes a chain iterator.

        This iterator returns runs from the first input paramter until it is exhausted,
        then proceeds to the next parameter, until all of the parameters are exhausted.
        Other parameters have default values if they don't be specified by additional
        key-value pairs.

        Args:
            params (dict, optional): Parameter range. Key is a parameter name and
                value is an iterable of parameter's value.
                use_best_param: If True (default), the parameter which got the best
                score is used during the following iterations.
            **kwargs: Additional parameter range.

        See Also:
            [Chain section](/tutorial/task#chain) in Multiple Runs Tutorial
        """
        params, base_params = utils.params.split_params(params, **kwargs)
        if self.tracking:
            self.tracking.set_tags(self.id, params)
            self.tracking.set_tags(self.id, base_params)
        params_list = {arg: list(value) for arg, value in params.items()}
        total = sum(len(value) for value in params_list.values())
        if "verbose" in base_params and base_params["verbose"] == 0:
            bar = None
        else:
            bar = tqdm(total=total, desc="Chain")
        best_params: Dict[str, Any] = {}
        for arg, values in params_list.items():
            best_param = None
            for value in values:
                args = base_params.copy()
                args.update({arg: value})
                if use_best_param:
                    args.update(best_params)
                run = self.create_run(args)
                yield run
                if run.monitor:
                    current_score = run.monitor.best_score
                    if best_param is None:
                        best_score = current_score
                        best_param = value
                    elif run.monitor.mode == "min" and current_score < best_score:
                        best_score = current_score
                        best_param = value
                    if current_score > best_score:
                        best_score = current_score
                        best_param = value
                del run
                gc.collect()
            if best_param is not None:
                best_params[arg] = best_param
            if bar is not None:
                bar.update(1)
        self.terminate()


class Study(Task):
    """Study class create a parent run to manage hyperparameter tuning."""

    def optimize(self, suggest_name: str = "", **kwargs):
        """Performs parameter optimizations using Optuna.

        Args:
            suggest_name: Name of suggest function.
            **kwargs: Key-iterable pairs for parametric optimization.

        See Also:
            [Hyperparameter Tuning](/tutorial/tuning) in Tutorial
        """

        if not suggest_name:
            suggest_name = list(self.objective.suggests.keys())[0]
        study_name = ".".join([self.experiment_name, suggest_name, self.name])
        mode = self.create_instance("monitor").mode
        study = self.tuner.create_study(study_name, mode)
        if self.tracking:
            self.tracking.set_tags(self.id, {"study_name": study_name})
            study.set_user_attr("run_id", self.id)
        has_pruning = self.tuner.pruner is not None
        optimize_args = inspect.signature(study.optimize).parameters.keys()
        params = {}
        for key in list(kwargs.keys()):
            if key not in optimize_args:
                params[key] = kwargs.pop(key)
        create_run = functools.partial(self.create_run, **params)
        objective = self.objective(suggest_name, create_run, has_pruning)
        study.optimize(objective, **kwargs)
        self.terminate()
        return study
