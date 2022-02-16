"""
Functions related to detecting the text of standard objects.
"""
from curses.ascii import isupper
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
    Need way to filter out object color from text color.

    Ideas
    -----
    kmeans to filter down to most common color in bounds
        - likely to be the color of the text
    get average color after kmeans

    Parameters
    ----------
    img : np.ndarray
        the image the text is in
    bounds : np.ndarray
        bounds of the text

    Returns
    -------
    np.ndarray
        the color of the text
    """
    # kmeans to get single color
    cropped_img = img[bounds[0, 0] : bounds[2, 0], bounds[0, 1] : bounds[2, 1]]
    # kmeans_img = cv2.kmeans(cropped_img, K=1,)

    # get average color of detected text
    color = np.array(
        [
            np.mean(img[:, :, 0]),
            np.mean(img[:, :, 1]),
            np.mean(img[:, :, 2]),
        ]  ## TODO: swtich to kmeans image
    )

    # map detected color to available colors in competition
    ## TODO: need to get way to correlate to available competition colors

    return color


def detect_text(img: np.ndarray, bounds: np.ndarray = None) -> np.ndarray:
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
    np.ndarray
        np.ndarray containing detected characters, format: ([bounds], 'character', color)

    ## TODO: Documentation to be updated later
    """
    # correct image if necessary
    corrected_img = img
    if bounds != None:
        corrected_img = splice_rotate_img(img, bounds)

    # Image processing to make text more clear
    gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, ksize=(5, 5), sigmaX=0)
    
    # cv2.imshow("Blurred", blurred_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # laplace_img = cv2.Laplacian(blurred_img, ddepth=cv2.CV_8U, ksize=5)

    # cv2.imshow("Laplacian", laplace_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_image = np.dstack((blurred_img, blurred_img, blurred_img))

    print("Image processing complete.")
    # detect text
    txt_data = pytesseract.image_to_data(
        blurred_img, output_type=pytesseract.Output.DICT, lang="eng", config='--psm 10'
    )
    print(txt_data)
    # filter detected text to find valid characters
    found_characters = []
    for i, txt in enumerate(txt_data["text"]):
        # Shows all detected text ## TODO: remove
        # print("Text:", txt)
        # x = txt_data["left"][i]
        # y = txt_data["top"][i]
        # w = txt_data["width"][i]
        # h = txt_data["height"][i]
        # bounds = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        # for j in range(4):
            # cv2.line(output_image, bounds[j], bounds[j+1], (0, 255, 0), thickness=2)

        if (txt != None) and (len(txt) == 1):  # length of 1
            # must be uppercase letter or number
            if (txt.isalpha() and isupper(txt)) or txt.isnumeric():
                # get data for each text object detected
                x = txt_data["left"][i]
                y = txt_data["top"][i]
                w = txt_data["width"][i]
                h = txt_data["height"][i]

                bounds = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                color = 'Green' # TODO: remove
                # color = get_text_color(img, bounds) # TODO: uncomment when implemented

                # add to found characters array
                found_characters += [(txt, bounds, color)]
    
    for c in found_characters:
        cv2.line(output_image, c[1][0], c[1][1], (0, 0, 255), thickness=2)
        cv2.line(output_image, c[1][1], c[1][2], (0, 0, 255), thickness=2)
        cv2.line(output_image, c[1][2], c[1][3], (0, 0, 255), thickness=2)
        cv2.line(output_image, c[1][3], c[1][0], (0, 0, 255), thickness=2)

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return found_characters


if __name__ == "__main__":
    """
    Driver for testing text detection and classification functions.
    """
    img = cv2.imread("/home/cameron/Documents/GitHub/SUAS-2022/vision/Object Detection/text_y.jpg")

    detected_chars = detect_text(img)
    
    print("The following characters were found in the image:", detected_chars)