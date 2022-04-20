"""
Functions relating to detection of the emergent object.
"""

from typing import List, Tuple

import cv2

import numpy as np
import numpy.typing as npt

from vision.common.bounding_box import BoundingBox, ObjectType


def get_emg_bounds(img: npt.NDArray[np.uint8]) -> List[BoundingBox]:
    """
    Gets the bounds of the emergent object in an image.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the image to find the emergent object in
    """
    # Initialize HOG object
    hog: cv2.HOGDescriptor = cv2.HOGDescriptor()  # the hog
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Find boudning boxes
    win_stride: Tuple[int, int] = (4, 4)  # sliding window step size x, y
    padding: Tuple[int, int] = (8, 8)  # x, y padding
    scale: float = 1.05  # determines size of image pyramid

    rects: List[Tuple[int, int, int, int]]
    rects, _ = hog.detectMultiScale(img, winStride=win_stride, padding=padding, scale=scale)

    # Convert to bounding boxes
    detected_people: List[BoundingBox] = []
    for x, y, width, height in rects:
        bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        )
        box = BoundingBox(vertices=bounds, obj_type=ObjectType.EMG_OBJECT)
        detected_people.append(box)
    return detected_people


# Driver for testing emergent object functions
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Runs emergent object detection algorithms. Must specify a file."
    )

    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="Filename of the image to run on. Required argument.",
    )

    args: argparse.Namespace = parser.parse_args()

    # no file name specified, cannot continue
    if not args.file_name:
        raise RuntimeError("No file specified.")
    file_name: str = args.file_name

    # Read in the image
    test_img: npt.NDArray[np.uint8] = cv2.imread(file_name)
