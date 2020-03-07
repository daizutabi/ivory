from dataclasses import dataclass, field
from typing import List


class Callback:
    def on_fit_start(self, cfg):
        pass

    def on_epoch_start(self, cfg):
        pass

    def on_train_start(self, cfg):
        pass

    def on_train_end(self, cfg):
        pass

    def on_val_start(self, cfg):
        pass

    def on_val_end(self, cfg):
        pass

    def on_epoch_end(self, cfg):
        pass

    def on_fit_end(self, cfg):
        pass


@dataclass
class CallbackCaller:
    callbacks: List[Callback] = field(default_factory=list, repr=False)

    def call(self, name: str, cfg):
        for callback in [cfg.metrics] + self.callbacks:
            getattr(callback, name)(cfg)

    def on_fit_start(self, cfg):
        self.call("on_fit_start", cfg)

    def on_epoch_start(self, cfg):
        self.call("on_epoch_start", cfg)

    def on_train_start(self, cfg):
        self.call("on_train_start", cfg)

    def on_train_end(self, cfg):
        self.call("on_train_end", cfg)

    def on_val_start(self, cfg):
        self.call("on_val_start", cfg)

    def on_val_end(self, cfg):
        self.call("on_val_end", cfg)

    def on_epoch_end(self, cfg):
        self.call("on_epoch_end", cfg)

    def on_fit_end(self, cfg):
        self.call("on_fit_end", cfg)
