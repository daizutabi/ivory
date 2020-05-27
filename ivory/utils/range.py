import numpy as np


class Range:
    def __init__(self, start, stop, step=1, num: int = 0, log: bool = False):
        self.start = start
        self.stop = stop
        self.step = step
        self.num = num
        self.log = log
        if log and step != 1:
            raise ValueError(f"Invalid step.")

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
        if self.num >= 2:
            s += f", n={self.num}"
        return s + ")"

    def __iter__(self):
        if self.is_integer:
            if self.start < self.stop:
                it = range(self.start, self.stop + 1, self.step)
            else:
                it = range(self.start, self.stop - 1, -self.step)
            if self.num < 2:
                return iter(it)
            else:
                values = list(it)
                index = np.linspace(0, len(values) - 1, self.num)
                return (values[int(round(x))] for x in index)
        else:
            num = self.num
            if self.log:
                if self.num < 2:
                    raise ValueError(f"num must be larger than 1, but {num} given.")
                start = np.log10(self.start)
                stop = np.log10(self.stop)
                return iter(float(x) for x in np.logspace(start, stop, num))
            else:
                if num < 2:
                    num = round(abs(self.stop - self.start) / self.step + 1)
                return iter(float(x) for x in np.linspace(self.start, self.stop, num))

    def __len__(self):
        return len(list(iter(self)))
