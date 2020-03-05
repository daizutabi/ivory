from dataclasses import dataclass
from typing import Callable, Optional

import hydra
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


def create_data(num_samples=10000):
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

    def __call__(self, input):
        return (input + self.std * np.random.randn(2)).astype(np.float32)


@dataclass
class DataLoaders(DataFrameLoaders):
    def __post_init__(self):
        self.input = ["x", "y"]
        self.target = ["z"]


class Model(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Metrics(ivory.torch.Metrics):
    pass


class Trainer(ivory.torch.Trainer):
    def train_step(self, model, input):
        return model(input)

    def validate_step(self, model, input):
        return model(input)


class Objective:
    def __init__(self, min_x=-10, max_x=10):
        self.min_x = min_x
        self.max_x = max_x

    def __call__(self, trial):
        x = trial.suggest_uniform("x", self.min_x, self.max_x)
        return (x - 2) ** 2


@hydra.main(config_path="config.yaml")
def main(config):
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    cfg = ivory.utils.parse(config)
    print(cfg.trainer)
    train_loader, val_loader = cfg.dataloaders[0]
    print(len(cfg.data))
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    cfg.data

    cfg.trainer.fit(train_loader, val_loader, cfg)

    cfg.metrics.dataframe(columns=["a"])

    cfg.data.iloc[9]

    # study = optuna.create_study()
    # study.optimize(runner.objective, n_trials=3, callbacks=[callback])


if __name__ == "__main__":
    main()
