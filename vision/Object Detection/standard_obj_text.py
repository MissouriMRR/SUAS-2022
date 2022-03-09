"""
Algorithms related to detecting the text of standard objects.
"""

import cv2
from cv2 import kmeans
import numpy as np
import pytesseract

# Possible colors
colors = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (127, 127, 127),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (),
    "PURPLE": (),
    "BROWN": (),
    "ORANGE": (),
    "BLACK": (),
}


class TextCharacteristics:
    """
    Class for detecting characteristics of text on standard objects.
    Characteristics consist of the character, orientation, and color.
    """

    def __init__(self):
        self.rotated_img = np.array([])

    def get_text_characteristics(self, img: np.ndarray, bounds: np.ndarray) -> tuple:
        """
        Gets the characteristics of the text on the standard object.

        Parameters
        ----------
        img : np.ndarray
            image to find characteristics of text within
        bounds : np.ndarray
            bounds of standard object containing text on in image

        Returns
        -------
        tuple
            characteristics of the text in the form (character, orientation, color)
        """
        ## Get the character ##
        characters = self._detect_text(img, bounds)
        # if len(characters) != 1: ## TODO: REVERT
        #     return (None, None, None)
        # character, char_bounds = characters[0]

        ## Get the orientation ##
        # orientation = self._get_orientation(img, char_bounds)

        ## Get the color of the text ##
        # color = self._get_text_color(img, char_bounds)
        color = self._get_text_color(img, bounds)  ## TODO: REVERT CHANGES HERE

        return (character, orientation, color)

    def _detect_text(self, img: np.ndarray, bounds: np.ndarray) -> list:
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
        list
            list containing detected characters and their bounds, Format of characters is ([bounds], 'character').
        """
        ## Crop and rotate the image ##
        rotated_img = self._slice_rotate_img(img, bounds)

        ## Image preprocessing to make text more clear ##
        processed_img = self._preprocess_img(rotated_img)
        output_image = np.dstack((processed_img, processed_img, processed_img))

        ## Detect Text ##
        txt_data = pytesseract.image_to_data(
            output_image,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 10",
        )

        ## Filter detected text to find valid characters ##
        found_characters = []
        for i, txt in enumerate(txt_data["text"]):
            if (txt != None) and (len(txt) == 1):  # length of 1
                # must be uppercase letter or number
                if (txt.isalpha() and txt.isupper()) or txt.isnumeric():
                    # get data for each text object detected
                    x = txt_data["left"][i]
                    y = txt_data["top"][i]
                    w = txt_data["width"][i]
                    h = txt_data["height"][i]

                    # Don't continue processing if text is size of full image
                    img_h, img_w = np.shape(processed_img)
                    if not (x == 0 and y == 0 and w == img_w and h == img_h):
                        bounds = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                        # add to found characters array
                        found_characters += [(txt, bounds)]

        return found_characters

    def _get_text_color(self, img: np.ndarray, char_bounds: np.ndarray) -> str:
        """
        Detect the color of the text.

        Parameters
        ----------
        img : np.ndarray
            the image the text is in
        char_bounds : np.ndarray
            bounds of the text

        Returns
        -------
        str
            the color of the text
        """
        CHAR_BOUNDS = [
            [94, 42],
            [94, 140],
            [162, 140],
            [162, 42],
        ]  ## TODO: TEMP: change to function parameter

        # Slice image around bounds of text ##
        x_vals = [coord[0] for coord in CHAR_BOUNDS]
        y_vals = [coord[1] for coord in CHAR_BOUNDS]
        min_x = np.amin(x_vals)
        max_x = np.amax(x_vals)
        min_y = np.amin(y_vals)
        max_y = np.amax(y_vals)

        cropped_img = self.rotated_img[
            min_y:max_y, min_x:max_x, :
        ]  ## TODO: fix crop for text?

        ## Image preprocessing to make text bigger/clearer ##
        blur = cv2.medianBlur(cropped_img, ksize=9)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated = cv2.dilate(erosion, kernel=kernel, iterations=1)

        ## KMeans with k=2 to seperate object and text color ##
        
        # Convert to (R, G, B, X, Y)
        vectorized = dilated.reshape((-1, 3))
        idxs = np.array([idx for idx, _ in np.ndenumerate(np.mean(dilated, axis=2))])
        vectorized = np.hstack((vectorized, idxs))

        # Run Kmeans
        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        _, label, center = cv2.kmeans(np.float32(vectorized), K=K, bestLabels=None, criteria=term_crit, attempts=10, flags=0)
        center = np.uint8(center)[:, :3]

        # Convert back to RGB
        kmeans_img = center[label.flatten()]
        kmeans_img = kmeans_img.reshape((dilated.shape))

        cv2.imshow("Kmeans Img", kmeans_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ## Determine which of the 2 colors is more central ##
        img_colors = np.unique(kmeans_img, axis=1)
        color_1_mat = np.where(
            kmeans_img == img_colors[0], 255, 128
        )  ## TODO: check this
        color_2_mat = np.where(kmeans_img == img_colors[1], 255, 128)

        dimensions = np.shape(kmeans_img)
        center_pt = (
            int(dimensions[0] / 2),
            int(dimensions[1] / 2),
        )
        color_1_mat[center_pt] = 0
        color_2_mat[center_pt] = 0

        dist_1_mat = cv2.distanceTransform(color_1_mat, cv2.DIST_L2, 3)
        dist_1 = cv2.mean(
            dist_1_mat, kmeans_img == img_colors[0]
        )  ## TODO: Check if only need 1 calc for this
        dist_2_mat = cv2.distanceTransform(color_1_mat, cv2.DIST_L2, 3)
        dist_2 = cv2.mean(
            dist_2_mat, kmeans_img == img_colors[1]
        )  ## TODO: Check if only need 1 calc for this

        color = min(dist_1, dist_2)

        ## Match found color to color enum ##

        color = "Green"
        return color

    def _get_orientation(self, img: np.ndarray, bounds: np.ndarray) -> str:
        """
        Get the orientation of the text.

        Parameters
        ----------
        img : np.ndarray
            the image the text is in
        bounds : np.ndarray
            bounds of the text
        """
        ## TEMP: Implemmentation out of scope of current issue
        orientation = "N"

        return orientation

    def _slice_rotate_img(self, img: np.ndarray, bounds: np.ndarray) -> np.ndarray:
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
        ## Slice image around bounds and find center point ##
        x_vals = [coord[0] for coord in bounds]
        y_vals = [coord[1] for coord in bounds]
        min_x = np.amin(x_vals)
        max_x = np.amax(x_vals)
        min_y = np.amin(y_vals)
        max_y = np.amax(y_vals)

        cropped_img = img[min_x:max_x][min_y:max_y][:]

        dimensions = np.shape(cropped_img)
        center_pt = (
            int(dimensions[0] / 2),
            int(
                dimensions[1] / 2,
            ),
        )

        ## Get angle of rotation ##
        ## TODO: 1st index depends on how bounds stored for standard object
        tl_x = bounds[0][0]
        tr_x = bounds[3][0]
        tl_y = bounds[0][1]
        tr_y = bounds[3][1]
        angle = np.rad2deg(np.arctan((tr_y - tl_y) / (tr_x - tl_x)))

        ## Rotate image ##
        rot_mat = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
        self.rotated_img = cv2.warpAffine(
            cropped_img, rot_mat, cropped_img.shape[1::-1], flags=cv2.INTER_LINEAR
        )

        return self.rotated_img

    def _preprocess_img(self, img: np.ndarray) -> np.ndarray:
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

        # erode and dilate to increase text clarity and reduce noise
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel=kernel, iterations=1)

        # laplace edge detection
        laplace_img = cv2.Laplacian(dilated, ddepth=cv2.CV_8U, ksize=5)

        # binarize image
        binarized = np.where(laplace_img > 50, np.uint8(255), np.uint8(0))

        # additional blur to remove noise
        blur_2 = cv2.medianBlur(binarized, ksize=3)

        return blur_2


if __name__ == "__main__":
    """
    Driver for testing text detection and classification functions.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Runs text characteristics algorithms. Must specify a file."
    )

    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="Filename of the image. Required argument.",
    )

    args = parser.parse_args()

    # no benchmark name specified, cannot continue
    if not args.file_name:
        raise RuntimeError("No file specified.")
    file_name = args.file_name

    img = cv2.imread(file_name)

    # bounds for stock image, given by standard object detection
    ## TODO: Change to function once implemented
    bounds = [[77, 184], [3, 91], [120, 0], [194, 82]]

    detector = TextCharacteristics()
    detected_chars = detector.get_text_characteristics(img, bounds)

    print("The following character was found in the image:", detected_chars)
