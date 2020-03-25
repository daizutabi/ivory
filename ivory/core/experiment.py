from ivory.core.base import Base


class Experiment(Base):
    __slots__ = []  # type:ignore

    def set_client(self, client):
        if client.tracker:
            self.set_tracker(client.tracker)
        if client.tuner:
            self.set_tuner(client.tuner)

    def set_tracker(self, tracker):
        self.set(tracker=tracker)
        if not self.name:
            self.name = "Default"
            self.params["experiment"]["name"] = self.name
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            self.params["experiment"]["id"] = self.id

    def set_tuner(self, tuner):
        self.set(tuner=tuner)
