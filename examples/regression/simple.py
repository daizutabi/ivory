from dataclasses import dataclass

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame

import ivory
import ivory.torch
from ivory.torch.data import DataFrameLoaders
from ivory.utils import kfold_split


def create_data(num_samples=1000):
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    df = DataFrame(xy, columns=["x", "y"])
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    df["z"] = ((df.x + dx) * (df.y + dy)).astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df


@dataclass
class Transform:
    std: float = 0

    def __call__(self, input, target, mode="train"):
        if mode == "train":
            input = (input + self.std * np.random.randn(2)).astype(np.float32)
        return input, target


@dataclass(repr=False)
class DataLoaders(DataFrameLoaders):
    def __post_init__(self):
        self.input = ["x", "y"]
        self.target = ["z"]


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


class Metrics(ivory.torch.Metrics):
    def evaluate(self, loss, output, target):
        mse = torch.mean((output - target) ** 2).item()
        return {"loss": loss, "mse": mse}


class Trainer(ivory.torch.Trainer):
    pass


class Run(ivory.torch.Run):
    pass


class Objective(ivory.Objective):
    def __call__(self, trial):
        num_layers = trial.suggest_int("num_layers", 1, 3)
        for i in range(num_layers):
            trial.suggest_int(f"model.hidden_sizes.{i}", 5, 30)
        run = self.create_run(trial.params)
        run.start(fold=0)
        return run.metrics.best_score


def main():
    objective = ivory.create_objective("config.yaml")
    # run = objective.create_run()
    objective.set_default(["data"])
    storage = "mysql+mysqldb://daizu:tabi@localhost/optuna"
    study_name = "example-study"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    study.optimize(objective, n_trials=3, n_jobs=1)

    study.best_params
    objective.config(study.best_params)

    # study.trials_dataframe()
    # optuna.visualization.plot_optimization_history(study)


if __name__ == "__main__":
    main()
