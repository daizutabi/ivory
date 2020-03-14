from collections import abc


class Callback:
    methods = ["on_fit_start", "on_epoch_start", "on_epoch_end", "on_fit_end"]


class CallbackCaller(abc.Mapping):
    __slots__ = ["callbacks"]

    def __init__(self, callbacks=None):
        objects = []
        for key in self:
            if isinstance(self[key], Callback):
                objects.append(self[key])
        if callbacks is not None:
            objects += callbacks
        self.callbacks = {}
        for method in Callback.methods:
            objs = [o for o in objects if hasattr(o, method)]
            methods = [getattr(o, method) for o in objs]

            def callback(methods=methods):
                for method in methods:
                    method(self)

            callback.objects = objs
            self.callbacks[method] = callback
