class Base:
    __slots__ = ["id", "name", "params", "objects"]

    def __init__(self, params, **objects):
        self.id = self.name = ""
        if "id" in objects:
            self.id = objects.pop("id")
        if "name" in objects:
            self.name = objects.pop("name")
        self.params = params
        self.objects = objects

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


CALLBACK_METHODS = [
    "on_fit_start",
    "on_epoch_start",
    "on_train_start",
    "on_train_end",
    "on_val_start",
    "on_val_end",
    "on_epoch_end",
    "on_fit_end",
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
