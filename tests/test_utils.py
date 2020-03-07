import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import ivory.utils as U


def test_kfold_split():
    x = np.arange(100)
    fold = U.kfold_split(x, 5)
    assert fold.min() == 0
    assert fold.max() == 4
    assert len(fold[fold == 3]) == 20


def test_parse_single(config_single):
    cfg = U.parse(config_single)
    assert isinstance(cfg.data, np.ndarray)
    assert cfg.data[0] == 1


def test_parse_multi(config):
    cfg = U.parse(config)
    assert isinstance(cfg.series, pd.Series)
    assert cfg.series[0] == 1

    config[1].series.data = "$.data"
    cfg = U.parse(config)
    assert cfg.series[0] == 1
    assert len(cfg.series) == 2

    config[1].series.data = "$.data.shape"
    cfg = U.parse(config)
    assert cfg.series[0] == 2
    assert len(cfg.series) == 1


def test_parse_default(config):
    cfg1 = U.parse(config)
    cfg2 = U.parse(config)
    assert cfg1.data is not cfg2.data and cfg1.series is not cfg2.series
    cfg2 = U.parse(config, default={"data": cfg1.data})
    assert cfg1.data is cfg2.data and cfg1.series is not cfg2.series
    cfg2 = U.parse(config, default=cfg1)
    assert cfg1.data is cfg2.data and cfg1.series is cfg2.series


def test_parse_keys(config):
    cfg = U.parse(config, keys=["data"])
    assert "data" in cfg and "series" not in cfg


def test_instantiate(config):
    data = U.instantiate(config, "data")
    assert isinstance(data, np.ndarray)
    series = U.instantiate(config, "series", default={"data": data})
    assert isinstance(series, pd.Series)


def test_parse_extra():
    config = OmegaConf.create([{"data": {"a": 1, "b": 2}}, {"x": 100}])
    cfg = U.parse(config)
    assert isinstance(cfg.data, DictConfig)
    assert cfg.data["a"] == 1 and cfg.data.b == 2
    assert cfg.x == 100
