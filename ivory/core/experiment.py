from ivory.core.base import Creator


class Experiment(Creator):
    def set_tracker(self, tracker):
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            class_name = self.__class__.__name__.lower()
            self.params[class_name]["id"] = self.id
        self["tracker"] = tracker

    def create_task(self):
        return self.create_run(name="task")

    def create_study(self):
        return self.create_run(name="study")
