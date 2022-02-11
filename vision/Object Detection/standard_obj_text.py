"""
Functions related to detecting the text of standard objects.
"""
import cv2
from matplotlib.pyplot import get
import numpy as np
import pytesseract


def splice_rotate_img(img: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Splice a portion of an image and rotate to be rectangular.

    Parameters
    ----------
    img : np.ndarray
        the image to take a splice of
    bounds : np.ndarray
        array of tuple bounds (4 x-y coordinates)

    Returns
    -------
    np.ndarray
        the spliced/rotated images
    """
    return


def get_text_color(img: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Detect the color of the text.
    Will return interop_api_pb2.Odlc.%COLOR% for odlc.alphanumeric_color
    ## TODO: change to return type to interop enum class
    """
    return


def detect_text(img: np.ndarray, bounds: np.ndarray = None) -> tuple:
    """
    Detect text within an image.
    Will return string for parameter odlc.alphanumeric

    Parameters
    ----------
    img : np.ndarray
        image to detect text within
    bounds : np.ndarray
        array of tuple bounds (4 x-y coordinates)

    Returns
    -------
    tuple
        tuple containing bounds array and the character detected ([bounds], 'character', color)

    ## TODO: Documentation to be updated later
    """
    corrected_img = img
    if bounds != None:
        corrected_img = splice_rotate_img(img, bounds)

    txt_data = pytesseract.image_to_data(
        img_rgb, output_type=pytesseract.Output.DICT, lang="eng"
    )  # get data from image

    color = get_text_color()

    return


"""
Random Ideas:
- seperate detection functions
    - detect general - detection for when drone is just flying around
    - detect on object - detection for when object has been identified, will crop/rotate image to encompass the object and then detect text
- multiprocess image on each of 4 axis for text
    - would be quicker and check multiple angles for the text
    - would need function to map detected text back to unrotated images
- text color
    - find average color of area within detected text and map to nearest color in interop
    OR
    - check for each color from interop within detected text - slower
"""

if __name__ == "__main__":
    """
    Driver for testing text detection and classification functions.
    """
    img = cv2.imread("test_image.jpg")
    img_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # Convert to RGB ## TODO: remove, not needed for camera feed, only for opencv imreads
