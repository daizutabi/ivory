import copy
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {}

DEFAULTS["client"] = {"client": {"tracker": {}}}
DEFAULTS["experiment"] = {"experiment": {}}
DEFAULTS["task"] = {"task": {}}
DEFAULTS["study"] = {"study": {"objective": {}, "tuner": {}}}


def get(name: str) -> Dict[str, Any]:
    return copy.deepcopy(DEFAULTS[name])


DEFAULT_CLASS: Dict[str, Any] = {}

DEFAULT_CLASS["core"] = {
    "client": "ivory.core.client.Client",
    "tracker": "ivory.core.tracker.Tracker",
    "tuner": "ivory.core.tuner.Tuner",
    "experiment": "ivory.core.base.Experiment",
    "objective": "ivory.core.objective.Objective",
    "run": "ivory.core.run.Run",
    "task": "ivory.core.run.Task",
    "study": "ivory.core.run.Study",
    "dataset": "ivory.core.data.Dataset",
    "datasets": "ivory.core.data.Datasets",
    "results": "ivory.callbacks.results.Results",
    "metrics": "ivory.callbacks.metrics.Metrics",
    "monitor": "ivory.callbacks.monitor.Monitor",
    "early_stopping": "ivory.callbacks.early_stopping.EarlyStopping",
}

DEFAULT_CLASS["torch"] = {
    "run": "ivory.torch.run.Run",
    "dataset": "ivory.torch.data.Dataset",
    "results": "ivory.torch.results.Results",
    "metrics": "ivory.torch.metrics.Metrics",
    "trainer": "ivory.torch.trainer.Trainer",
}

DEFAULT_CLASS["tensorflow"] = {
    "run": "ivory.tensorflow.run.Run",
    "trainer": "ivory.tensorflow.trainer.Trainer",
}

DEFAULT_CLASS["nnabla"] = {
    "results": "ivory.callbacks.results.BatchResults",
    "metrics": "ivory.nnabla.metrics.Metrics",
    "trainer": "ivory.nnabla.trainer.Trainer",
}


DEFAULT_CLASS["sklearn"] = {
    "estimator": "ivory.sklearn.estimator.Estimator",
    "metrics": "ivory.sklearn.metrics.Metrics",
}


def update_class(params: Dict[str, Any], library: str = "core"):
    if "library" in params:
        library = params.pop("library")
    for key, value in params.items():
        if value is None:
            value = {}
            params[key] = value
        if not isinstance(value, dict):
            continue
        if "library" in value:
            library = value.pop("library")
        for kind in ["class", "def", "call"]:
            if kind in value:
                attr = value[kind]
                break
        else:
            kind = "class" if key != "dataset" else "def"
            if key in DEFAULT_CLASS[library]:
                attr = DEFAULT_CLASS[library][key]
            elif key in DEFAULT_CLASS["core"]:
                attr = DEFAULT_CLASS["core"][key]
            else:
                attr = None
            if attr:
                value[kind] = attr
            else:
                params[key] = None
                break

        from ivory.core import instance

        attr = instance.get_attr(attr)
        if "__requires__" in dir(attr):
            requires = getattr(attr, "__requires__")
            for r in requires:
                if r not in value:
                    value[r] = {}
        update_class(value, library)
