"""
Functions relating to detection of the emergent object.
"""


from typing import List, Optional, Tuple

import cv2

import numpy as np
import numpy.typing as npt

from vision.common.bounding_box import BoundingBox, ObjectType


def preprocess_emg_obj(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Preprocessing to increase accuracy of emergent object detection.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the image containing the emergent object

    Returns
    -------
    resized : npt.NDArray[np.uint8]
        the preprocessed image
    """

    # Resize image to increase accuracy and reduce processing time
    width: int = min(400, img.shape[1])
    height: int = int(width * (img.shape[0] / img.shape[1]))
    resized: npt.NDArray[np.uint8] = cv2.resize(
        img, (width, height), interpolation=cv2.INTER_LINEAR
    )

    return resized


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

    # Run hog people detection
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


def crop_emg_obj(img: npt.NDArray[np.uint8], bbox: BoundingBox) -> npt.NDArray[np.uint8]:
    """
    Crops the image containing the emergent object to the BoundingBox.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        the image containing the emergent object
    bbox : BoundingBox
        the bounding box containing the detected emergent object

    Returns
    -------
    cropped_img : npt.NDArray[np.uint8]
        the image cropped around the bounding box
    """
    cropped_img: npt.NDArray[np.uint8] = np.copy(img)

    x_min: int
    x_max: int
    x_min, x_max = bbox.get_x_extremes()
    y_min: int
    y_max: int
    y_min, y_max = bbox.get_y_extremes()

    cropped_img = cropped_img[y_min:y_max, x_min:x_max]

    return cropped_img


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
    processed_img: npt.NDArray[np.uint8] = preprocess_emg_obj(test_img)

    cv2.imshow("Preprocessed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    emg_box: Optional[BoundingBox] = get_emg_bounds(processed_img)

    if emg_box is not None:
        print(emg_box)

        # Show the found bounding box
        result_img: npt.NDArray[np.uint8] = np.copy(test_img)
        min_x: int
        max_x: int
        min_x, max_x = emg_box.get_x_extremes()
        min_y: int
        max_y: int
        min_y, max_y = emg_box.get_y_extremes()
        cv2.rectangle(result_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        cv2.imshow("Detected Object", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Crop the image around the object and show
        obj_img: npt.NDArray = crop_emg_obj(test_img, emg_box)

        cv2.imshow("Cropped Image", obj_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No Object Found")
