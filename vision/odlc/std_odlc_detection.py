"""
Algorithms relating to the detection of the standard odlc objects.
"""

from typing import Any, Dict, List, Tuple, Optional

import cv2

import numpy as np
import numpy.typing as npt

from vision.common.bounding_box import ObjectType, BoundingBox


def preprocess_odlc() -> None:
    """
    Preprocess image for use in odlc object detection.
    """
    raise NotImplementedError("ODLC object detection not implemented")


def odlc_detection() -> None:
    """
    Detects odlc objects in the image.
    """
    raise NotImplementedError("ODLC object detection not implemented")


def find_odlc_objs() -> None:
    """
    Runs odlc detection algorithms to find the odlc objects in the image.
    """
    raise NotImplementedError("ODLC object detection not implemented")


# Driver for testing algoirhtms relating to odlc object detection
if __name__ == "__main__":
    pass
