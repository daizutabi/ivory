from collections import abc


class Callback:
    @classmethod
    def on_experiment_start(cls, experiment):
        pass

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


class CallbackCaller(abc.Mapping):
    __slots__ = ["callbacks"]

    def __init__(self, callbacks=None):
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def call(self, callback_name: str):
        for key in self:
            if isinstance(self[key], Callback):
                getattr(self[key], callback_name)(self)
        for callback in self.callbacks:
            getattr(callback, callback_name)(self)

    def on_fit_start(self):
        self.call("on_fit_start")

    def on_epoch_start(self):
        self.call("on_epoch_start")

    def on_train_start(self):
        self.call("on_train_start")

    def on_train_end(self):
        self.call("on_train_end")

    def on_val_start(self):
        self.call("on_val_start")

    def on_val_end(self):
        self.call("on_val_end")

    def on_epoch_end(self):
        self.call("on_epoch_end")

    def on_fit_end(self):
        self.call("on_fit_end")
