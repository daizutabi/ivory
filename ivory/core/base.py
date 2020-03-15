class Base:
    __slots__ = ["id", "name", "params", "objects"]

    def __init__(self, name, params, **objects):
        self.id = ""
        self.name = name
        self.params = params
        self.objects = objects

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(id='{self.run_id}', name='{self.name}', "
        s += f"num_objects={len(self)})"
        return s

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


CALLBACK_METHODS = ["on_fit_start", "on_epoch_start", "on_epoch_end", "on_fit_end"]


class CallbackCaller(Base):
    __slots__ = ["callbacks"]

    def create_callbacks(self):
        self.callbacks = {}
        for method in CALLBACK_METHODS:
            methods = []
            for key in self:
                if hasattr(self[key], method):
                    methods.append(getattr(self[key], method))

            def callback(methods=methods):
                for method in methods:
                    method(self)

            self.callbacks[method] = callback
