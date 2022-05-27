"""
Algorithms relating to the detection of the standard odlc objects.
"""

from typing import Any, Dict, List, Tuple, Optional

import cv2

import numpy as np
import numpy.typing as npt

# from vision.common.bounding_box import ObjectType, BoundingBox


def preprocess_odlc(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Preprocess image for use in odlc object detection.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the original image

    Returns
    -------
    preprocessed : npt.NDArray[np.uint8]
        the image after preprocessing
    """
    # grayscale
    gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # blur to remove noise
    blur: npt.NDArray[np.uint8] = cv2.medianBlur(gray, ksize=9)

    # threshold binarization
    threshold: npt.NDArray[np.uint8]
    _, threshold = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    return threshold


def odlc_detection() -> None:
    """
    Detects odlc objects in the image.
    """
    raise NotImplementedError("ODLC object detection not implemented")


def parse_contours() -> None:
    """
    Parses contours to create bounding boxes.
    """
    raise NotImplementedError("ODLC object detection not implemented")


def find_odlc_objs() -> None:
    """
    Runs odlc detection algorithms to find the odlc objects in the image.
    """
    raise NotImplementedError("ODLC object detection not implemented")


# Driver for testing algoirhtms relating to odlc object detection
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Runs text characteristics algorithms. Must specify a file."
    )

    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="Filename of the image. Required argument.",
    )

    args: argparse.Namespace = parser.parse_args()

    # no benchmark name specified, cannot continue
    if not args.file_name:
        raise RuntimeError("No file specified.")
    file_name: str = args.file_name

    test_img: npt.NDArray[np.uint8] = cv2.imread(file_name)

    cv2.imshow("Original Image", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    preprocessed_img = preprocess_odlc(test_img)

    cv2.imshow("Image after preprocessing", preprocessed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
