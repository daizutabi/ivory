from ivory.core.base import Base
from ivory.core.experiment import create_experiment
from ivory.core.instance import create_base_instance


class Environment(Base):
    __slots__ = []  # type:ignore

    def create_experiment(self, params, source_name=""):
        source_name = source_name or self.source_name
        experiment = create_experiment(params, source_name=source_name)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        if self.tuner:
            experiment.set_tuner(self.tuner)
        return experiment


def create_environment(params, source_name=""):
    return create_base_instance(params, "environment", source_name=source_name)
