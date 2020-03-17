import ivory
from ivory.core.base import Base


class Environment(Base):
    __slots__ = []  # type:ignore

    def create_experiment(self, params):
        experiment = ivory.create_experiment(params)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        if self.tuner:
            experiment.set_tuner(self.tuner)
        return experiment
