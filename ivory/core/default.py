import copy
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {}

DEFAULTS["client"] = {"client": {"tracker": {}}}
DEFAULTS["experiment"] = {"experiment": {}}
DEFAULTS["task"] = {"task": {}}


def get(name: str):
    return copy.deepcopy(DEFAULTS[name])


DEFAULT_CLASS: Dict[str, Any] = {}

DEFAULT_CLASS["core"] = {
    "client": "ivory.core.client.Client",
    "tracker": "ivory.core.tracker.Tracker",
    "tuner": "ivory.core.tuner.Tuner",
    "experiment": "ivory.core.experiment.Experiment",
    "objective": "ivory.core.objective.Objective",
    "run": "ivory.core.run.Run",
    "task": "ivory.core.run.Task",
    "study": "ivory.core.run.Study",
    "dataset": "ivory.core.data.Dataset",
    "datasets": "ivory.core.data.Datasets",
    "dataloaders": "ivory.core.data.DataLoaders",
    "results": "ivory.callbacks.results.Results",
    "metrics": "ivory.callbacks.metrics.Metrics",
    "monitor": "ivory.callbacks.monitor.Monitor",
    "early_stopping": "ivory.callbacks.early_stopping.EarlyStopping",
}

DEFAULT_CLASS["torch"] = {
    "run": "ivory.torch.run.Run",
    "dataloaders": "ivory.torch.data.DataLoaders",
    "dataset": "ivory.torch.data.Dataset",
    "results": "ivory.torch.results.Results",
    "metrics": "ivory.torch.metrics.Metrics",
    "trainer": "ivory.torch.trainer.Trainer",
}


def update_class(params, library="core"):
    for key, value in params.items():
        if value is None:
            value = {}
            params[key] = value
        if not isinstance(value, dict):
            continue
        if "library" in params[key]:
            library = params[key].pop("library")
        if "class" not in value and "def" not in value and "call" not in value:
            kind = "class" if key != "dataset" else "def"
            if key in DEFAULT_CLASS[library]:
                params[key][kind] = DEFAULT_CLASS[library][key]
            elif key in DEFAULT_CLASS["core"]:
                params[key][kind] = DEFAULT_CLASS["core"][key]
        update_class(value, library)
