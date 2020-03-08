from dataclasses import dataclass, field
from typing import List


class Callback:
    def on_fit_start(self, obj):
        pass

    def on_epoch_start(self, obj):
        pass

    def on_train_start(self, obj):
        pass

    def on_train_end(self, obj):
        pass

    def on_val_start(self, obj):
        pass

    def on_val_end(self, obj):
        pass

    def on_epoch_end(self, obj):
        pass

    def on_fit_end(self, obj):
        pass


@dataclass
class CallbackCaller:
    callbacks: List[Callback] = field(default_factory=list, repr=False)

    def call(self, name: str, obj):
        for callback in [obj.metrics] + self.callbacks:
            getattr(callback, name)(obj)

    def on_fit_start(self, obj):
        self.call("on_fit_start", obj)

    def on_epoch_start(self, obj):
        self.call("on_epoch_start", obj)

    def on_train_start(self, obj):
        self.call("on_train_start", obj)

    def on_train_end(self, obj):
        self.call("on_train_end", obj)

    def on_val_start(self, obj):
        self.call("on_val_start", obj)

    def on_val_end(self, obj):
        self.call("on_val_end", obj)

    def on_epoch_end(self, obj):
        self.call("on_epoch_end", obj)

    def on_fit_end(self, obj):
        self.call("on_fit_end", obj)
