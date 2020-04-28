from ivory.core.base import Creator


class Experiment(Creator):
    def set_tracker(self, tracker):
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            class_name = self.__class__.__name__.lower()
            self.params[class_name]["id"] = self.id
        self["tracker"] = tracker

    def create_task(self):
        return self.create_run(class_name="Task")

    def create_study(self, run_number: int = 0):
        return self.create_run(class_name="Study", run_number=run_number)

    def update_params(self, **default):
        self.tracker.update_params(self.id, **default)
