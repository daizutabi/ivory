from dataclasses import dataclass


@dataclass(repr=False)
class Dict:
    def __post_init__(self):
        self.dict = {}

    def __len__(self):
        return len(self.dict)

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:
            return Missing(self, key)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __contains__(self, key):
        return key in self.dict

    def __iter__(self):
        return iter(self.dict)

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
