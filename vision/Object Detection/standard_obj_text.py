"""
Functions related to detecting the text of standard objects.
"""
import cv2
import numpy as np
import pytesseract

def detect_text(img):
    """
    Detect text within an image.
    Will return string for parameter odlc.alphanumeric

    Parameters
    ----------
    img : np.ndarray
        image to detect text within

    Returns location and character. ## TODO: Documentation to be updated later
    """

    txt_data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT, lang="eng") # get data from image

    return


def get_text_color():
    """
    Detect the color of the text.
    Will return interop_api_pb2.Odlc.%COLOR% for odlc.alphanumeric_color
    """
    return


if __name__ == "__main__":
    """
    Driver for testing text detection and classification functions.
    """
    img = cv2.imread("test_image.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB ## TODO: remove, not needed for camera feed, only for opencv imreads

