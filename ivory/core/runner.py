from dataclasses import dataclass, field
from typing import List

from ivory.core.instance import Map, instantiate


@dataclass
class Runner:
    config: List[Map] = field(default_factory=list)

    def run(self, fold: int = 0):
        raise NotImplementedError


def create_runner(config: List[Map], default: Map = None):
    cfg = instantiate(config, default=default)
    assert "runner" in cfg and "run" not in cfg
    runner = cfg["runner"]
    for key in cfg:
        if key != "runner":
            setattr(runner, key, cfg[key])
    runner.config = config
    return runner
