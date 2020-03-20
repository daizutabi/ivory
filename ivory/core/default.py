DEFAULT_CLASS = {}

DEFAULT_CLASS["core"] = {
    "environment": "ivory.core.environment.Environment",
    "tracker": "ivory.core.tracker.Tracker",
    "tuner": "ivory.core.tuner.Tuner",
    "experiment": "ivory.core.experiment.Experiment",
    "objective": "ivory.core.objective.Objective",
    "monitor": "ivory.callbacks.monitor.Monitor",
    "early_stopping": "ivory.callbacks.early_stopping.EarlyStopping",
}

DEFAULT_CLASS["torch"] = {
    "run": "ivory.torch.run.Run",
    "dataloader": "ivory.torch.data.DataLoader",
    "metrics": "ivory.torch.metrics.Metrics",
    "trainer": "ivory.torch.trainer.Trainer",
}


def update_class(params, library=None):
    if "environment" in params:
        if "class" not in params["environment"]:
            params["environment"]["class"] = DEFAULT_CLASS["core"]["environment"]
        update_class(params["environment"])
    if "experiment" in params:
        if "class" not in params["experiment"]:
            params["experiment"]["class"] = DEFAULT_CLASS["core"]["experiment"]
        update_class(params["experiment"])
    if "run" in params:
        if "library" in params["run"]:
            library = params["run"].pop("library")
        else:
            library = None
        if "class" not in params["run"]:
            if library:
                params["run"]["class"] = DEFAULT_CLASS[library]["run"]
            else:
                raise ValueError("Can't find class for run.")
        update_class(params["run"], library)
    else:
        for key, value in params.items():
            if isinstance(value, dict) and "class" not in value:
                if library and key in DEFAULT_CLASS[library]:
                    params[key]["class"] = DEFAULT_CLASS[library][key]
                elif key in DEFAULT_CLASS["core"]:
                    params[key]["class"] = DEFAULT_CLASS["core"][key]
                else:
                    raise ValueError(f"Can't find class for {key}.")
