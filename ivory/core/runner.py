from ivory.core.instance import Map, instantiate


class Runner:
    def __init__(self, config: Map, default: Map = None):
        self.config = config
        objects = instantiate(config, default=default)
        for key in objects:
            setattr(self, key, objects[key])

    def __len__(self):
        return len(self.config)

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return iter(self.config)

    def __getitem__(self, key):
        return getattr(self, key)

    def run(self, fold: int = 0):
        raise NotImplementedError
