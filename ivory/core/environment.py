from ivory.core import instance
from ivory.core.base import Base
from ivory.core.experiment import create_experiment


class Environment(Base):
    __slots__ = []  # type:ignore

    def create_experiment(self, params="params.yaml"):
        experiment = create_experiment(params)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        if self.tuner:
            experiment.set_tuner(self.tuner)
        return experiment


create_environment = instance.create_instance_factory("environment")
