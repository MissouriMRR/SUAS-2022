"""
Functions related to detecting the text of standard objects.
"""
from curses.ascii import isupper
import cv2
from matplotlib.pyplot import get
import numpy as np
import pytesseract


def slice_rotate_img(img: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Slice a portion of an image and rotate to be rectangular.

    Parameters
    ----------
    img : np.ndarray
        the image to take a splice of
    bounds : np.ndarray
        array of tuple bounds (4 x-y coordinates; tl-tr-br-bl)

    Returns
    -------
    np.ndarray
        the spliced/rotated images
    """
    # Find center point
    min_x = np.amin(bounds[:][0])
    max_x = np.amax(bounds[:][0])
    min_y = np.amin(bounds[:][1])
    max_y = np.amax(bounds[:][1])
    center_pt = (int((max_x + min_x) / 2), int((max_y + min_y) / 2))
    print(center_pt)
    # Get angle of rotation
    tl_x = bounds[0][0]
    tr_x = bounds[1][0]
    tl_y = bounds[0][1]
    tr_y = bounds[1][1]
    angle = np.rad2deg(np.arctan((tr_y - tl_y) / (tr_x - tl_x)))
    print(angle)

    # Rotate image
    rot_mat = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.imshow("Rotated image", rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotated_img


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


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for text detection.

    Parameters
    ----------
    img : np.ndarray
        image to preprocess

    Returns
    -------
    np.ndarray
        the image after preprocessing
    """
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # blur to remove noise
    blur = cv2.medianBlur(gray, ksize=9)

    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # erode and dilate to increase text clarity and reduce noise
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(blur, kernel=kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel=kernel, iterations=1)

    laplace_img = cv2.Laplacian(dilated, ddepth=cv2.CV_8U, ksize=5)

    # binarize image
    binarized = np.where(laplace_img > 50, np.uint8(255), np.uint8(0))
    print(type(binarized[0][0]))
    # print(np.shape(binarized))

    # edge detection
    # edges = cv2.Canny(laplace_img, 100, 200)

    return binarized
    # return binarized


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
        corrected_img = slice_rotate_img(img, bounds)

    # Image processing to make text more clear
    processed_img = preprocess_img(corrected_img)

    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # output_image = np.dstack((blurred_img, blurred_img, blurred_img))
    output_image = np.dstack((processed_img, processed_img, processed_img))

    print("Image processing complete.")
    # detect text
    txt_data = pytesseract.image_to_data(
        output_image, output_type=pytesseract.Output.DICT, lang="eng", config="--psm 10"
    )
    print(txt_data)
    # filter detected text to find valid characters
    found_characters = []
    for i, txt in enumerate(txt_data["text"]):
        if (txt != None) and (len(txt) == 1):  # length of 1
            # must be uppercase letter or number
            if txt.isalpha() or txt.isnumeric():
                # if (txt.isalpha() and isupper(txt)) or txt.isnumeric():
                # get data for each text object detected
                x = txt_data["left"][i]
                y = txt_data["top"][i]
                w = txt_data["width"][i]
                h = txt_data["height"][i]

                bounds = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                color = "Green"  # TODO: remove
                # color = get_text_color(img, bounds) # TODO: uncomment when implemented

                # add to found characters array
                found_characters += [(txt, bounds, color)]

    ## TODO: convert bound coordinates back to regular image if rotated

    # Draw bounds of detected character
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
    img = cv2.imread(
        "/home/cameron/Documents/GitHub/SUAS-2022/vision/Object Detection/text_y.jpg"
    )
    # img = cv2.imread(
    #     "/home/cameron/Documents/GitHub/SUAS-2022/vision/Object Detection/letter_a.jpg"
    # )

    bounds = [[14, 63], [112, 5], [192, 231], [94, 173]]

    detected_chars = detect_text(img, bounds)

    print("The following characters were found in the image:", detected_chars)
