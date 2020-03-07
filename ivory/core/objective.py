from dataclasses import dataclass, field
from typing import List

from ivory.core.instance import Map, instantiate


@dataclass
class Objective:
    config: List[Map]
