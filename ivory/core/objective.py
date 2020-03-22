import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from optuna.trial import Trial

from ivory import utils


@dataclass
class Objective:
    suggest: Callable
    mode: str = "min"
    params: Dict[str, Any] = field(default_factory=dict, init=False)

    @utils.autoload
    def set_params(self, params, source_name):
        if "run" in params:
            params = params["run"]
        self.params = params

    def create_params(self, trial: Trial):
        self.suggest(trial)
        params = copy.deepcopy(self.params)
        utils.update_dict(params, trial.params)
        params["name"] = f"trial#{trial.number}"
        return params
