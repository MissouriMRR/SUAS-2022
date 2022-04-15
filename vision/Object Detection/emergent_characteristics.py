"""
Functions relating to the emergent object's characteristics.
"""


import os
import pathlib

import cv2

import numpy as np
import numpy.typing as npt


EMG_OBJECT_OUT_FOLDER = "emg_object"
EMG_IMG_FILE_NAME = "emg_object_img.jpg"
EMG_TEXT_FILE_NAME = "emg_object_description.txt"


def save_emg_img(img: npt.NDArray[np.uint8]) -> None:
    """
    Save the image of the emergent object and create a text file for
    the user to add a description.

    Parameters
    ----------
    img - npt.NDArray[np.uint8]
        the image containing the emergent object
    """
    # Find the paths
    path: pathlib.Path = pathlib.Path().resolve()
    output_folder = os.path.join(path, EMG_OBJECT_OUT_FOLDER)
    img_file: str = os.path.join(output_folder, EMG_IMG_FILE_NAME)
    description_file: str = os.path.join(output_folder, EMG_TEXT_FILE_NAME)

    # Create the output folder if necessary
    ## NOTE: handle if img/txt files exist already?
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save the image to the output file
    cv2.imwrite(filename=img_file, img=img)

    # Create the description file
    with open(description_file, "w", encoding="utf-8") as file:
        file.write("Provide a single-line description of the emergent object on the line below:\n")


def get_emg_description() -> str:
    """
    Reads the description of the emergent object from the text file.

    Returns
    -------
    description - str
        a description of the emergent object
    """
    raise NotImplementedError("get_emg_description() is not yet implemented")


# Driver for testing emergent object functions
if __name__ == "__main__":
    IMG_NAME = "test_img.jpg"
    print(IMG_NAME)
