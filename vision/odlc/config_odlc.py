"""
Constants to be used by ODLC algorithms.
"""

from typing import Dict, List

import numpy as np
import numpy.typing as npt


# Possible colors and HSV upper/lower bounds
POSSIBLE_COLORS: Dict[str, npt.NDArray[np.int64]] = {
    "WHITE": np.array([[180, 18, 255], [0, 0, 231]]),
    "BLACK": np.array([[180, 255, 30], [0, 0, 0]]),
    "GRAY": np.array([[180, 18, 230], [0, 0, 40]]),
    "RED": np.array(
        [[180, 255, 255], [159, 50, 70], [9, 255, 255], [0, 50, 70]]
    ),  # red wraps around and needs 2 ranges
    "BLUE": np.array([[128, 255, 255], [90, 50, 70]]),
    "GREEN": np.array([[89, 255, 255], [36, 50, 70]]),
    "YELLOW": np.array([[35, 255, 255], [25, 50, 70]]),
    "PURPLE": np.array([[158, 255, 255], [129, 50, 70]]),
    "BROWN": np.array([[20, 255, 180], [10, 100, 120]]),
    "ORANGE": np.array([[24, 255, 255], [10, 50, 70]]),
}


# Possible orientations of the odlc text
POSSIBLE_ORIENTATIONS: List[str] = [
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
]
