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
            args.append(f"id='{self.id}'")
        if self.name:
            args.append(f"name='{self.name}'")
        args.append(f"num_objects={len(self)}")
        args = ", ".join(args)
        return f"{self.__class__.__name__}({args})"


CALLBACK_METHODS = [
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


class CallbackCaller(Base):
    def create_callback(self):
        for method in CALLBACK_METHODS:
            methods = []
            for key in self:
                if hasattr(self[key], method):
                    callback = getattr(self[key], method)
                    if callable(callback):
                        methods.append(callback)

            def callback(methods=methods):
                for method in methods:
                    method(self)

            self[method] = callback
