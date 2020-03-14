import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame

import ivory
from ivory.utils import kfold_split


def create_data(num_samples=1000):
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    df = DataFrame(xy, columns=["x", "y"])
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    df["z"] = ((df.x + dx) * (df.y + dy)).astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df[["x", "y"]], df[["fold", "z"]]


class Model(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        for in_features, out_features in zip([2] + hidden_sizes, hidden_sizes + [1]):
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def objective(trial):
    trial.suggest_loguniform("optimizer.lr", 1e-5, 1e-1)


def optimize():
    experiment = ivory.create_experiment("params.yaml")



    experiment.start()

    experiment.tracking._c._tracking_client.tracking_uri

    experiment.tracking._c.get_experiment(experiment.experiment_id).artifact_location

    experiment.optuna.study.user_attrs


    experiment
    experiment.tracking

    run1 = experiment.create_run()
    run2 = experiment.create_run()

    run1.on_fit_end()

    from ivory.callbacks import Tracking
    tracking1 = Tracking()
    tracking1.on_fit_start(run1)
    tracking2 = Tracking()
    tracking2.on_fit_start(run2)

    run2.run_id
    tracking2.on_fit_end(run2)

    client = experiment.tracking.client
    r = client.create_run(experiment.experiment_id)

    client.log_artifacts(r.info.run_id, "C://Users/daizu/desktop/tmp")
    r.info

    client.log_metric(r.info.run_id, 'a', 1.2)


def main():
    optimize()


if __name__ == "__main__":
    main()
