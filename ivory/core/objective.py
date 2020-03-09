from dataclasses import dataclass, field
from typing import List

import yaml

from ivory.core.instance import Map, get_attr, instantiate
from ivory.core.run import Run
from ivory.utils import dot_to_list, to_float, update_dict


@dataclass
class Objective:
    run: str  # class name of Run to be created.
    yaml: str = field(default="", repr=False)
    default: Map = field(default_factory=dict, repr=False)

    def config(self, update: Map = None) -> Map:
        """Return a newly created config dictionary for each run.

        Config is always created from yaml string when this method is called.
        """
        config = to_float(yaml.safe_load(self.yaml))
        if update is None:
            return config
        else:
            update_dict(config, dot_to_list(update))
            return config

    def set_default(self, names: List[str]):
        """Set default objects which are shared for every run."""
        config = self.config()
        self.default = instantiate(config, names=names)

    def create_run(self, update: Map = None) -> Run:
        """Create a run for an optinal update config."""
        config = self.config(update)
        config.pop("objective")
        cls = get_attr(self.run)
        return cls(config, default=self.default)


def create_objective(yaml_config_file: str) -> Objective:
    """Create an Objective instance from a yaml config file.

    Parameters
    ----------
    yaml_config_file : str
        Yaml config file path.

    Returns
    -------
    Objective instance.
    """
    with open(yaml_config_file) as file:
        yml = file.read()
    config = to_float(yaml.safe_load(yml))
    objective = instantiate(config["objective"])
    objective.yaml = yml
    return objective
