import numpy as np
import optuna
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
    pruning = ivory.callbacks.Pruning(trial, "val_loss")
    run = ivory.create_run(trial.params, callbacks=[pruning])
    run.name = f"#{trial.number}"
    run.start()
    trial.set_user_attr("run_id", run.tracking.run_id)
    return run.metrics.best_score


def optimize():
    experiment = ivory.create_experiment("params.yaml")

    import inspect
    import optuna

def main():
    optimize()


if __name__ == "__main__":
    main()
