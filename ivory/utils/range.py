import ast
import re

import numpy as np


class Range:
    def __init__(self, start, stop=None, step=1, num: int = 0, log: bool = False):
        if isinstance(start, str):
            start, stop, step, num, log = parse(start)
        elif stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step
        self.num = num
        self.log = log
        if log and step != 1:
            raise ValueError("Invalid step.")

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
        if self.log:
            s += f", log={self.log}"
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


def parse(value: str):
    """
    Examples:
        >>> parse('2-3')
        (2, 3, 1, 0, False)
        >>> parse('2-4-2')
        (2, 4, 2, 0, False)
        >>> parse('2-4:3')
        (2, 4, 1, 3, False)
        >>> parse('0-1-0.2')
        (0, 1, 0.2, 0, False)
        >>> parse('0-1:5')
        (0, 1, 1, 5, False)
        >>> parse('0-1.log')
        (0, 1, 1, 0, True)
        >>> parse('1e-3_1e-2.log')
        (0.001, 0.01, 1, 0, True)
    """
    num = 0
    log = False
    if value.endswith(".log"):
        value = value[:-4]
        log = True
    if ":" in value:
        value, num_str = value.split(":")
        num = int(num_str)
    sep = "_" if "_" in value else "-"
    match = re.match(f"(.+){sep}(.+)", value)
    if not match:
        raise ValueError(f"Invalid string for Range: {value}")
    if sep in match.group(1):
        if num != 0:
            raise ValueError(f"Invalid string for Range: {value}")
        start, stop = match.group(1).split(sep)
        step = match.group(2)
    else:
        start = match.group(1)
        stop = match.group(2)
        step = "1"
    start = literal_eval(start)
    stop = literal_eval(stop)
    step = literal_eval(step)
    if all(isinstance(x, (int, float)) for x in [start, stop, step]):
        return start, stop, step, num, log
    raise ValueError(f"Invalid string for Range: {value}")


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
