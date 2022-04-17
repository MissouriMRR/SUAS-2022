"""
Functions relating to the emergent object's characteristics.
"""


import os
import pathlib

import cv2

import numpy as np
import numpy.typing as npt


# File name constants
EMG_OBJECT_OUT_FOLDER: str = "emg_object"
EMG_IMG_FILE_NAME: str = "emg_object_img.jpg"
EMG_TEXT_FILE_NAME: str = "emg_object_description.txt"


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
    with open(description_file, mode="w", encoding="utf-8") as file:
        file.write("Provide a single-line description of the emergent object on the line below:\n")


def get_emg_description() -> str:
    """
    Reads the description of the emergent object from the text file.

    Returns
    -------
    description - str
        a description of the emergent object
    """
    # Find the paths
    path: pathlib.Path = pathlib.Path().resolve()
    description_file: str = os.path.join(path, EMG_OBJECT_OUT_FOLDER, EMG_TEXT_FILE_NAME)

    # description of the emergent object provided by the user
    description: str = ""

    # Get the description from the file
    with open(description_file, mode="r", encoding="utf-8") as file:
        i = 0
        for line in file:  # avoid file size errors
            if i == 1:  # only accept single line description
                description = line
            i += 1
    return description


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

    # Save image and create text file
    save_emg_img(test_img)

    # Wait until user adds description to text file
    input("Press enter once description has been added")

    # Read description of emergent object from text file
    print("Descrtiption:", get_emg_description())
