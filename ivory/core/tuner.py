from dataclasses import dataclass

import optuna


@dataclass
class Tuner:
    storage: str = "sqlite://"
    load_if_exists: bool = True

    def create_study(
        self, study_name: str, mode: str, experiment_id=None, sampler=None, pruner=None
    ):
        """Creates and returns a Optuna Study object."""
        if mode == "min":
            direction = "minimize"
        elif mode == "max":
            direction = "maximize"
        else:
            raise ValueError("monitor's mode must be 'min' or 'max'.")
        study = optuna.create_study(
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            direction=direction,
            load_if_exists=self.load_if_exists,
        )
        if experiment_id:
            study.set_user_attr("experiment_id", experiment_id)
        return study

    def delete_study(self, study_name: str):
        optuna.delete_study(storage=self.storage, study_name=study_name)
