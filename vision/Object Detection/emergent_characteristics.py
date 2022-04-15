"""
Functions relating to the emergent object's characteristics.
"""


import numpy as np
import numpy.typing as npt


def save_emg_img(img: npt.NDArray[np.uint8]) -> None:
    """
    Save the image of the emergent object and create a text file for
    the user to add a description.

    Parameters
    ----------
    img - npt.NDArray[np.uint8]
        the image containing the emergent object
    """
    raise NotImplementedError("save_emg_img() is not yet implemented")


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
