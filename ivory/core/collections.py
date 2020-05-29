class Dict:
    def __init__(self):
        self.dict = {}

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.dict)

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.dict[list(self.keys())[key]]
        else:
            return self.dict[key]

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            return Missing(self, key)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __contains__(self, key):
        return key in self.dict

    def __iter__(self):
        return iter(self.dict)

    def __dir__(self):
        return super().__dir__() + list(self.dict.keys())

    def __repr__(self):
        class_name = self.__class__.__name__
        args = ", ".join(f"{key!r}" for key in self.keys())
        return f"{class_name}([{args}])"

    def __call__(self, **kwargs):
        self.set(**kwargs)
        return self

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.dict[key] = value

    def update(self, *args, **kwargs):
        self.dict.update(*args, **kwargs)

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()

    def copy(self):
        return self.dict.copy()


class List:
    def __init__(self):
        self.list = []

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.list)

    def __getitem__(self, key):
        return self.list[key]

    def __contains__(self, value):
        return value in self.list

    def __iter__(self):
        return iter(self.list)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self.list})"

    def set(self, values):
        self.list = list(values)

    def append(self, value):
        self.list.append(value)

    def copy(self):
        return self.list.copy()


class Missing:
    def __init__(self, obj, key):
        self.obj = obj
        self.key = key

    def __bool__(self):
        return False

    def __getitem__(self, index):
        raise AttributeError(f"{self.obj} has not attribute '{self.key}'")

    def __getattr__(self, index):
        raise AttributeError(f"{self.obj} has not attribute '{self.key}'")
