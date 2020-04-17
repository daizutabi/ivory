from ivory.core.dict import Dict


class Base(Dict):
    def __init__(self, params, **objects):
        super().__init__()
        self.id = self.name = self.source_name = ""
        self.params = params
        if "id" in objects:
            self.id = objects.pop("id")
        if "name" in objects:
            self.name = objects.pop("name")
        if "source_name" in objects:
            self.source_name = objects.pop("source_name")
        self.dict = objects

    def __repr__(self):
        args = []
        if self.id:
            args.append(f"id={self.id!r}")
        if self.name:
            args.append(f"name={self.name!r}")
        args.append(f"num_objects={len(self)}")
        args = ", ".join(args)
        return f"{self.__class__.__name__}({args})"


class Callback:
    METHODS = [
        "on_init",
        "on_fit_start",
        "on_epoch_start",
        "on_train_start",
        "on_train_end",
        "on_val_start",
        "on_val_end",
        "on_epoch_end",
        "on_fit_end",
        "on_test_start",
        "on_test_end",
    ]

    def __init__(self, caller, methods):
        self.caller = caller
        self.methods = methods

    def __repr__(self):
        class_name = self.__class__.__name__
        callbacks = list(self.methods.keys())
        return f"{class_name}({callbacks})"

    def __call__(self):
        caller = self.caller
        for method in self.methods.values():
            method(caller)


class CallbackCaller(Base):
    def create_callback(self):
        for method in Callback.METHODS:
            methods = {}
            for key in self:
                if hasattr(self[key], method):
                    callback = getattr(self[key], method)
                    if callable(callback):
                        methods[key] = callback

            self[method] = Callback(self, methods)
