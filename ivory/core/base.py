"""This module provides base classes for Ivory."""
import copy
import inspect
from typing import Callable, Dict, Tuple

import ivory.core.collections
from ivory import utils
from ivory.core import default, instance


class Base(ivory.core.collections.Dict):
    """Base class for an entity class such as `Client`, `Experiment`, and `Run`.

    Args:
        params (dict, optional): Parameter dictionary to create this instance.
        **instances: Member instances. Key is its name and value is the member instance.

    Attributes:
        params (dict, optional): Parameter dictionary to create this instance.
        id (str): Instance ID given by
            [MLFlow Tracking](https://www.mlflow.org/docs/latest/tracking.html).
        name (str): Instance name.
        source_name (str): Name of the YAML parameter file that is used to create
            this instance.
    """

    def __init__(self, params=None, **instances):
        super().__init__()
        self.params = params
        self.id = self.name = self.source_name = ""
        if "id" in instances:
            self.id = instances.pop("id")
        if "name" in instances:
            self.name = instances.pop("name")
        if "source_name" in instances:
            self.source_name = instances.pop("source_name")
        self.dict = instances

    def __repr__(self):
        args = []
        if self.id:
            args.append(f"id={self.id!r}")
        if self.name:
            args.append(f"name={self.name!r}")
        args.append(f"num_instances={len(self)}")
        args = ", ".join(args)
        return f"{self.__class__.__name__}({args})"


class Creator(Base):
    """Creator class to create `Run` instances."""

    @property
    def experiment_id(self) -> str:
        return self.params["experiment"]["id"]

    @property
    def experiment_name(self) -> str:
        return self.params["experiment"]["name"]

    def create_params(
        self, args=None, name: str = "run", **kwargs
    ) -> Tuple[dict, dict]:
        """Returns a tuple of (parameter dictionary, update dictionary).

        The parameter dictionary is deeply copied from original one, then updated
        according to the arguments. The update dictionary includes updated parameter
        only.

        Args:
            args (dict, optional): Update dictionary.
            name: Run class name in lower case.
            **kwargs: Additional update dictionary.

        Examples:
            Use `args` for parameters including dots:

                params, update = experiment.create_params(
                    {'hidden_sizes.0': 100}, fold=3
                )

            The `params` is the full parameter dictionary. while the `update` is a
            part of `params`, i.e., `update = {'hidden_sizes.0': 100, 'fold': 3}`.
        """
        params = copy.deepcopy(self.params)
        if name not in params:
            params.update(default.get(name))
        update, args = utils.params.create_update(params[name], args, **kwargs)
        utils.params.update_dict(params[name], update)
        return params, args

    def create_run(self, args=None, name: str = "run", **kwargs):
        """Creates a `Run` instance according to arguments.

        Args:
            args (dict, optional): Update dictionary.
            name: Run class name in lower case.
            **kwargs: Additional update dictionary.

        Returns:
            Run: Created `Run` instance. The parameter for this instance is the
                returned dictionary from the
                [`create_params()`](#ivory.core.base.Creator.create_params) method.
        """
        params, args = self.create_params(args, name, **kwargs)
        run = instance.create_base_instance(params, name, self.source_name)
        if self.tracker:
            from ivory.callbacks.pruning import Pruning

            run.set_tracker(self.tracker, name)
            run.tracking.log_params_artifact(run)
            args = {arg: utils.params.get_value(run.params[name], arg) for arg in args}
            run.tracking.log_params(run.id, args)
            run.set(pruning=Pruning())
        return run

    def create_instance(self, instance_name: str, args=None, name="run", **kwargs):
        """Creates an member instance of a `Run` according to arguments.

        Args:
            instance_name: Name of a member instance to create.
            args (dict, optional): Update dictionary.
            name: Run class name in lower case.
            **kwargs: Additional update dictionary.

        Returns:
            Created instance. The parameter for this instance is the
                returned directory from the
                [`create_params()`](#ivory.core.base.Creator.create_params) method.
        """
        params, _ = self.create_params(args, name, **kwargs)
        return instance.create_instance(params[name], instance_name)


class Callback:
    """Callback class for the Ivory callback system."""

    METHODS = [
        "on_init_begin",
        "on_init_end",
        "on_fit_begin",
        "on_epoch_begin",
        "on_train_begin",
        "on_train_end",
        "on_val_begin",
        "on_val_end",
        "on_epoch_end",
        "on_fit_end",
        "on_test_begin",
        "on_test_end",
    ]

    ARGUMENTS = ["run"]

    def __init__(self, caller: "CallbackCaller", methods: Dict[str, Callable]):
        self.caller = caller
        self.methods = methods

    def __repr__(self):
        class_name = self.__class__.__name__
        callbacks = list(self.methods.keys())
        return f"{class_name}({callbacks})"

    def __call__(self):
        caller = self.caller
        for method in self.methods.values():
            method(caller)


class CallbackCaller(Creator):
    """Callback caller class."""

    def create_callbacks(self):
        """Creates callback functions and store them in a dictionary."""
        for method in Callback.METHODS:
            methods = {}
            for key in self:
                if hasattr(self[key], method):
                    callback = getattr(self[key], method)
                    if callable(callback):
                        parameters = inspect.signature(callback).parameters
                        if list(parameters.keys()) == Callback.ARGUMENTS:
                            methods[key] = callback

            self[method] = Callback(self, methods)


class Experiment(Creator):
    """Experimet class, which is one of the main classes of Ivory library.

    Basically, one experiment is corresponding to one YAML parameter file that is held
    in an `Experiment` instance as a parameter dictionary. This parameter dictionary
    defines the default parameter values to create `Run` instances.

    See Also:
        The base class [`ivory.core.base.Creator`](#ivory.core.base.Creator)
        defines some methods to create a `Run` instance or its member instance.
    """

    def set_tracker(self, tracker):
        """Sets a `Tracker` instance for tracking.

        Args:
            tracker (Tracker): Tracker instance.
        """
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            self.params["experiment"]["id"] = self.id
        self.set(tracker=tracker)

    def create_task(self):
        """Creates a `Task` instance for multiple runs.

        See Also:
            For more details, see
            [client.create_task()](/api/ivory.core.client#ivory.core.client.Client.create_task)

            [Multiple Runs](/tutorial/task) in Tutorial.
        """
        return self.create_run(name="task")

    def create_study(self, args=None, **suggests):
        """Creates a `Study` instance for hyperparameter tuning.

        See Also:
            For more details, see
            [client.create_study()](/api/ivory.core.client#ivory.core.client.Client.create_study)

            [Hyperparameter Tuning](/tutorial/tuning) in Tutorial
        """
        study = self.create_run(name="study")
        if isinstance(args, str) and args in study.objective:
            study.objective.suggests = {args: study.objective.suggests[args]}
            return study
        if args or suggests:
            study.objective.update(args, **suggests)
        return study
