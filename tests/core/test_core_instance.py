import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ivory.core.instance import instantiate


def test_instantiate_single(config_single):
    cfg = instantiate(config_single)
    assert isinstance(cfg.data, np.ndarray)
    assert cfg.data[0] == 1


def test_parse_multi(config):
    cfg = instantiate(config)
    assert isinstance(cfg.series, pd.Series)
    assert cfg.series[0] == 1

    config[1].series.data = "$.data"
    cfg = instantiate(config)
    assert cfg.series[0] == 1
    assert len(cfg.series) == 2

    config[1].series.data = "$.data.shape"
    cfg = instantiate(config)
    assert cfg.series[0] == 2
    assert len(cfg.series) == 1


def test_parse_default(config):
    cfg1 = instantiate(config)
    cfg2 = instantiate(config)
    assert cfg1.data is not cfg2.data and cfg1.series is not cfg2.series
    cfg2 = instantiate(config, default={"data": cfg1.data})
    assert cfg1.data is cfg2.data and cfg1.series is not cfg2.series
    cfg2 = instantiate(config, default=cfg1)
    assert cfg1.data is cfg2.data and cfg1.series is cfg2.series


def test_parse_keys(config):
    cfg = instantiate(config, keys=["data"])
    assert "data" in cfg and "series" not in cfg


def test_parse_extra():
    config = OmegaConf.create([{"data": {"a": 1, "b": 2}}, {"x": 100}])
    cfg = instantiate(config)
    assert isinstance(cfg.data, DictConfig)
    assert cfg.data["a"] == 1 and cfg.data.b == 2
    assert cfg.x == 100
