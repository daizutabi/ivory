DEFAULT_CLASS = {}

DEFAULT_CLASS["core"] = {
    "client": "ivory.core.client.Client",
    "tracker": "ivory.core.tracker.Tracker",
    "tuner": "ivory.core.tuner.Tuner",
    "experiment": "ivory.core.experiment.Experiment",
    "objective": "ivory.core.objective.Objective",
    "run": "ivory.core.run.Run",
    "dataset": "ivory.core.data.Dataset",
    "results": "ivory.callbacks.results.Results",
    "monitor": "ivory.callbacks.monitor.Monitor",
    "early_stopping": "ivory.callbacks.early_stopping.EarlyStopping",
    "evaluator": "ivory.callbacks.evaluator.Evaluator",
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
            else:
                raise ValueError(f"Can't find class for {key}.")
        update_class(value, library)
