"""
Functions relating to detection of the emergent object.
"""

import cv2

import numpy as np
import numpy.typing as npt

from vision.common.bounding_box import BoundingBox


def get_emg_bounds(img: npt.NDArray[np.uint8]) -> BoundingBox:
    """
    Gets the bounds of the emergent object in an image.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the image to find the emergent object in
    """
    raise NotImplementedError("Function not implemented yet.")


# Driver for testing emergent object functions
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Runs emergent object characteristics algorithms. Must specify a file."
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
