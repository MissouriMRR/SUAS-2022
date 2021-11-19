"""
Find the bounding box of the contour and crop down the image to include only the object of interest.
"""
import cv2
import numpy as np
import argparse
import random


def get_bounding_image(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Calculate the bounding box with the given contour and then crop the given image.

    Parameters
    ----------
    img: numpy.array
        The input image to be cropped.
    contour: numpy.array
        The contour located in the given image. Used to find bounding box for cropping.

    Returns
    -------
    numpy.array - Cropped image.
    """
    # Find the bounding box of the contour.
    x, y, w, h = cv2.boundingRect(contour)
    # Return the cropped image using numpy splicing.
    return img[y : y + h, x : x + w]


if __name__ == "__main__":
    """
    Driver for testing object_bounding_box.
    """
    # Create object for parsing command-line arguments.
    parser = argparse.ArgumentParser(
        description="Read the given input image and crop based on a randomly generated contour."
    )
    # Add parser arguments.
    parser.add_argument("-i", "--input", type=str, help="Path to image file.")
    # Parse the command line arguments to an object.
    args = parser.parse_args()

    # Check if input was given.
    if args.input:
        # Open the img at the given path.
        try:
            img = cv2.imread(args.input)
        except FileNotFoundError as e:
            print(f"Unable to get image file: {e}")

        # Get image size info.
        h, w, c = img.shape
        # Generate a random contour, then find bounding box and crop the given image.
        contour = np.array(
            [
                [random.randint(0, w), random.randint(0, h)],
                [random.randint(0, w), random.randint(0, h)],
                [random.randint(0, w), random.randint(0, h)],
            ],
            dtype=np.int32,
        )

        # Call bounding box function.
        cropped = get_bounding_image(img, contour)
        # Display new image.
        cv2.imshow("bounding box", cropped)
        cv2.imshow("original", img)
        cv2.waitKey(0)
    else:
        # Throw error if no input file was given.
        raise FileNotFoundError(
            "No input image file has been given. For help type --help"
        )
