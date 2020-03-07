from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
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
        hidden_sizes = list(hidden_sizes)
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


class Runner(ivory.torch.Runner):
    pass


class Objective:
    def __init__(self, min_x=-10, max_x=10):
        self.min_x = min_x
        self.max_x = max_x

    def __call__(self, trial):
        x = trial.suggest_uniform("x", self.min_x, self.max_x)
        return (x - 2) ** 2


def main():
    config = OmegaConf.load("config.yaml")

    with open('config.yaml') as file:
        yml = yaml.safe_load(file)
        print(yml)


    print(yaml.dump(yml, default_flow_style=False, sort_keys=False))

    a = copy.deepcopy(config)
    a = copy.copy(config)

    a[0].data.num_samples = 10
    config[0].data
    {"data.num_samples": 10}

    cfg = ivory.core.instance.instantiate(config)
    cfg.runner.cfg

    runner = Runner.create(config)
    runner.cfg.model
    runner.run(fold=0)
    print(runner.cfg.metrics.best_result)
    print(runner.cfg.metrics.best_epoch)

    data = ivory.utils.instantiate(config, "data")

    cfg1 = ivory.utils.parse(config, {"data": data})
    cfg2 = ivory.utils.parse(config, {"data": data})
    cfg1.data is cfg2.data

    transform = ivory.utils.instantiate(config, "transform")

    cfg1 = ivory.utils.parse(config, keys=["data"])

    cfg2 = ivory.utils.parse(config, default=cfg1)

    cfg2.data is cfg1.data
    cfg2.transform is cfg1.transform

    data = runner.cfg.data
    id(data)
    runner2 = Runner.create(config, cfg1)
    runner3 = Runner.create(config, cfg1)

    runner2.cfg.data is runner3.cfg.data
    runner2.cfg.transform is runner3.cfg.transform

    # study = optuna.create_study()
    # study.optimize(runner.objective, n_trials=3, callbacks=[callback])


if __name__ == "__main__":
    main()
