import copy

from ivory import utils
from ivory.core import instance
from ivory.core.dict import Dict


class Base(Dict):
    def __init__(self, params, **objects):
        super().__init__()
        self.params = params
        self.id = self.name = self.source_name = ""
        if "id" in objects:
            self.id = objects.pop("id")
        if "name" in objects:
            self.name = objects.pop("name")
        if "source_name" in objects:
            self.source_name = objects.pop("source_name")
        self.dict = objects

    def __repr__(self):
        args = []
        if self.id:
            args.append(f"id={self.id!r}")
        if self.name:
            args.append(f"name={self.name!r}")
        args.append(f"num_objects={len(self)}")
        args = ", ".join(args)
        return f"{self.__class__.__name__}({args})"


class Creator(Base):
    @property
    def experiment_id(self):
        return self.params["experiment"]["id"]

    @property
    def experiment_name(self):
        return self.params["experiment"]["name"]

    def create_params(self, args=None, **kwargs):
        params = copy.deepcopy(self.params)
        update, args = utils.create_update(params["run"], args, **kwargs)
        utils.update_dict(params["run"], update)
        return params, args

    def create_run(self, args=None, class_name="Run", **kwargs):
        params, args = self.create_params(args, **kwargs)
        name = class_name.lower()
        if name not in params:
            params[name] = {}
        if self.tracker:
            run_name = self.tracker.create_run_name(self.experiment_id, class_name)
            params[name]["name"] = run_name
        run = instance.create_base_instance(params, name, self.source_name)
        if self.tracker:
            run.set_tracker(self.tracker)
            args = {arg: utils.get_value(run.params["run"], arg) for arg in args}
            run.tracking.log_params(run.id, args)
        return run

    def create_instance(self, name: str, args=None, **kwargs):
        params, _ = self.create_params(args, **kwargs)
        return instance.create_instance(params["run"], name)


class Callback:
    METHODS = [
        "on_init_start",
        "on_init_end",
        "on_fit_start",
        "on_epoch_start",
        "on_train_start",
        "on_train_end",
        "on_val_start",
        "on_val_end",
        "on_epoch_end",
        "on_fit_end",
        "on_test_start",
        "on_test_end",
    ]

    def __init__(self, caller, methods):
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
    def create_callbacks(self):
        for method in Callback.METHODS:
            methods = {}
            for key in self:
                if hasattr(self[key], method):
                    callback = getattr(self[key], method)
                    if callable(callback):
                        methods[key] = callback

            self[method] = Callback(self, methods)
