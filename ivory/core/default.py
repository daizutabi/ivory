DEFAULT_CLASS = {}

DEFAULT_CLASS["core"] = {
    "client": "ivory.core.client.Client",
    "tracker": "ivory.core.tracker.Tracker",
    "tuner": "ivory.core.tuner.Tuner",
    "experiment": "ivory.core.experiment.Experiment",
    "objective": "ivory.core.objective.Objective",
    "run": "ivory.core.run.Run",
    "results": "ivory.callbacks.results.Results",
    "monitor": "ivory.callbacks.monitor.Monitor",
    "early_stopping": "ivory.callbacks.early_stopping.EarlyStopping",
}

DEFAULT_CLASS["torch"] = {
    "run": "ivory.torch.run.Run",
    "dataloaders": "ivory.torch.data.DataLoaders",
    "results": "ivory.torch.results.Results",
    "metrics": "ivory.torch.metrics.Metrics",
    "trainer": "ivory.torch.trainer.Trainer",
}


def update_class(params, library="core"):
    for name in ["client", "experiment"]:
        if name in params:
            if "class" not in params[name]:
                params[name]["class"] = DEFAULT_CLASS["core"][name]
            update_class(params[name])
    if "run" in params:
        if "library" in params["run"]:
            library = params["run"].pop("library")
        if "class" not in params["run"]:
            params["run"]["class"] = DEFAULT_CLASS[library]["run"]
        update_class(params["run"], library)
    else:
        for key, value in params.items():
            if value is None:
                value = {}
                params[key] = value
            if isinstance(value, dict) and "class" not in value:
                if key in DEFAULT_CLASS[library]:
                    params[key]["class"] = DEFAULT_CLASS[library][key]
                elif key in DEFAULT_CLASS["core"]:
                    params[key]["class"] = DEFAULT_CLASS["core"][key]
                else:
                    raise ValueError(f"Can't find class for {key}.")
