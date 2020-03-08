from ivory.core.instance import Map, instantiate


class Runner:
    def __init__(self, config: Map, default: Map = None):
        cfg = instantiate(config, default=default)
        for key in cfg:
            setattr(self, key, cfg[key])
        self.config = config

    def run(self, fold: int = 0):
        raise NotImplementedError
