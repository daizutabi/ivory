import numpy as np


class Range:
    def __init__(self, start, stop, step=1, n: int = 0):
        self.start = start
        self.stop = stop
        self.step = step
        self.n = n

    @property
    def is_integer(self):
        return all(isinstance(x, int) for x in [self.start, self.stop, self.step])

    @property
    def is_float(self):
        return not self.is_integer

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}({self.start}, {self.stop}"
        if self.step != 1:
            s += f", {self.step}"
        if self.n >= 2:
            s += f", n={self.n}"
        return s + ")"

    def __iter__(self):
        if self.is_integer:
            if self.start < self.stop:
                it = range(self.start, self.stop + 1, self.step)
            else:
                it = range(self.start, self.stop - 1, -self.step)
            if self.n < 2:
                return iter(it)
            else:
                values = list(it)
                index = np.linspace(0, len(values) - 1, self.n)
                return (values[int(round(x))] for x in index)
        else:
            n = self.n
            if n < 2:
                n = round(abs(self.stop - self.start) / self.step + 1)
            return iter(float(x) for x in np.linspace(self.start, self.stop, n))

    def __len__(self):
        return len(list(iter(self)))
