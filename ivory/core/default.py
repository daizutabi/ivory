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
