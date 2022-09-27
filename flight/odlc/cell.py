from dataclasses import dataclass


@dataclass
class Cell:
    probability: float
    seen: bool
