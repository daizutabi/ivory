class Base:
    __slots__ = ["id", "name", "source_name", "params", "objects"]

    def __init__(self, params, **objects):
        self.id = self.name = self.source_name = ""
        self.params = params
        if "id" in objects:
            self.id = objects.pop("id")
        if "name" in objects:
            self.name = objects.pop("name")
        if "source_name" in objects:
            self.source_name = objects.pop("source_name")
        self.objects = objects

    def __repr__(self):
        args = []
        if self.id:
            args.append(f"id='{self.id}'")
        if self.name:
            args.append(f"name='{self.name}'")
        args.append(f"num_objects={len(self)}")
        args = ", ".join(args)
        return f"{self.__class__.__name__}({args})"

    def __len__(self):
        return len(self.objects)

    def __contains__(self, key):
        return key in self.objects

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, key):
        return self.objects[key]

    def __getattr__(self, key):
        if key in self.objects:
            return self.objects[key]

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.objects[key] = value


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
    __slots__ = []  # type:ignore

    def create_callbacks(self):
        for method in CALLBACK_METHODS:
            methods = []
            for key in self:
                if hasattr(self[key], method):
                    methods.append(getattr(self[key], method))

            def callback(methods=methods):
                for method in methods:
                    method(self)

            self.objects[method] = callback
