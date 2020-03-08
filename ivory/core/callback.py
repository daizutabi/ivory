from dataclasses import dataclass, field
from typing import List


class Callback:
    def on_fit_start(self, run):
        pass

    def on_epoch_start(self, run):
        pass

    def on_train_start(self, run):
        pass

    def on_train_end(self, run):
        pass

    def on_val_start(self, run):
        pass

    def on_val_end(self, run):
        pass

    def on_epoch_end(self, run):
        pass

    def on_fit_end(self, run):
        pass


@dataclass
class CallbackCaller:
    callbacks: List[Callback] = field(default_factory=list, repr=False)

    def call(self, name: str, run):
        for callback in [run.metrics] + self.callbacks + [run]:
            getattr(callback, name)(run)

    def on_fit_start(self, run):
        self.call("on_fit_start", run)

    def on_epoch_start(self, run):
        self.call("on_epoch_start", run)

    def on_train_start(self, run):
        self.call("on_train_start", run)

    def on_train_end(self, run):
        self.call("on_train_end", run)

    def on_val_start(self, run):
        self.call("on_val_start", run)

    def on_val_end(self, run):
        self.call("on_val_end", run)

    def on_epoch_end(self, run):
        self.call("on_epoch_end", run)

    def on_fit_end(self, run):
        self.call("on_fit_end", run)
