"""
Functions relating to detection of the emergent object.
"""


from typing import List, Optional, Tuple

import cv2

import numpy as np
import numpy.typing as npt

from vision.common.bounding_box import BoundingBox, ObjectType


def get_emg_bounds(img: npt.NDArray[np.uint8]) -> Optional[BoundingBox]:
    """
    Gets the bounds of the emergent object in an image.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the image to find the emergent object in

    Returns
    -------
    box : Optional[BoundingBox]
        The BoundingBox around the emergent object if it is found.
        Returns None if no emergent object found or more than one found.
    """
    # Initialize HOG object
    hog: cv2.HOGDescriptor = cv2.HOGDescriptor()  # the hog
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Run detection
    win_stride: Tuple[int, int] = (4, 4)  # sliding window step size x, y
    padding: Tuple[int, int] = (8, 8)  # x, y padding
    scale: float = 1.05  # determines size of image pyramid

    rects: List[Tuple[int, int, int, int]]
    rects, _ = hog.detectMultiScale(img, winStride=win_stride, padding=padding, scale=scale)

    # return none if no object found or more than one found
    if len(rects) != 1:
        return None

    # Convert to bounding boxes
    x: int
    y: int
    width: int
    height: int
    x, y, width, height = rects[0]
    bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )
    box = BoundingBox(vertices=bounds, obj_type=ObjectType.EMG_OBJECT)

    return box


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

    # Get the bounds in the image
    bbox: Optional[BoundingBox] = get_emg_bounds(test_img)
    print(bbox)

    if bbox is not None:
        # Show the found bounding box
        result_img: npt.NDArray[np.uint8] = np.copy(test_img)
        x_min: int
        x_max: int
        x_min, x_max = bbox.get_x_extremes()
        y_min: int
        y_max: int
        y_min, y_max = bbox.get_y_extremes()
        cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
