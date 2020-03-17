import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from optuna.trial import Trial

from ivory import utils
from ivory.utils import load_params


@dataclass
class Objective:
    suggest: Callable
    mode: str = "min"
    params: Dict[str, Any] = field(default_factory=dict, init=False)

    def set_params(self, params):
        if isinstance(params, str):
            params = load_params(params)
        if "run" in params:
            params = params["run"]
        self.params = params

    def create_params(self, trial: Trial):
        self.suggest(trial)
        params = copy.deepcopy(self.params)
        utils.update_dict(params, trial.params)
        params["name"] = f"trial#{trial.number}"
        return params
