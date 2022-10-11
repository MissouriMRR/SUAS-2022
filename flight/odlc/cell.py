from dataclasses import dataclass


@dataclass
class Cell:
    probability: float
    seen: bool
    x: float
    y: float
    is_valid: bool
