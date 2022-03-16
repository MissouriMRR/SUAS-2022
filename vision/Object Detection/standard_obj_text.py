"""
Algorithms related to detecting the text of standard objects.
"""

import cv2
from cv2 import kmeans
from matplotlib.pyplot import hsv
import numpy as np
import pytesseract


# Possible colors and HSV upper/lower bounds
POSSIBLE_COLORS = {
    "WHITE": np.array([[180, 18, 255], [0, 0, 231]]),
    "BLACK": np.array([[180, 255, 30], [0, 0, 0]]),
    "GRAY": np.array([[180, 18, 230], [0, 0, 40]]),
    "RED": np.array(
        [[180, 255, 255], [159, 50, 70], [9, 255, 255], [0, 50, 70]]
    ),  # red wraps around and needs 2 ranges
    "BLUE": np.array([[128, 255, 255], [90, 50, 70]]),
    "GREEN": np.array([[89, 255, 255], [36, 50, 70]]),
    "YELLOW": np.array([[35, 255, 255], [25, 50, 70]]),
    "PURPLE": np.array([[158, 255, 255], [129, 50, 70]]),
    "BROWN": np.array([[20, 255, 180], [10, 100, 120]]),
    "ORANGE": np.array([[24, 255, 255], [10, 50, 70]]),
}


class TextCharacteristics:
    """
    Class for detecting characteristics of text on standard objects.
    Characteristics consist of the character, orientation, and color.
    """

    def __init__(self):
        # related to text detection
        self.rotated_img = np.array([])

        # related to text color
        self.text_cropped_img = np.array([])
        self.kmeans = np.array([])

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
        if len(characters) != 1:
            return (None, None, None)
        character, char_bounds = characters[0]

        ## Get the orientation ##
        orientation = self._get_orientation(img, char_bounds)

        ## Get the color of the text ##
        color = self._get_text_color(img, char_bounds)

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
        # Slice rotated image around bounds of text ##
        x_vals = [coord[0] for coord in char_bounds]
        y_vals = [coord[1] for coord in char_bounds]
        min_x: int = np.amin(x_vals)
        max_x: int = np.amax(x_vals)
        min_y: int = np.amin(y_vals)
        max_y: int = np.amax(y_vals)

        self.text_cropped_img: np.ndarray = self.rotated_img[
            min_y:max_y, min_x:max_x, :
        ]

        ## Run Kmeans with K=2 ##
        self._run_kmeans()

        ## Determine which of the 2 colors is more central ##
        ## NOTE: val is in BGR due to kmeans return
        color_val: np.ndarray = self._get_color_value()

        ## Match found color to color enum ##
        color: str = self._parse_color(color_val)

        return color

    def _run_kmeans(self) -> None:
        """
        Run kmeans with K=2 for color classification.

        Returns
        -------
        None
        """
        ## Image preprocessing to make text bigger/clearer ##
        blur: np.ndarray = cv2.medianBlur(self.text_cropped_img, ksize=9)

        kernel: np.ndarray = np.ones((5, 5), np.uint8)
        erosion: np.ndarray = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated: np.ndarray = cv2.dilate(erosion, kernel=kernel, iterations=1)

        ## Color and Location-based KMeans clustering ##

        # Convert to (R, G, B, X, Y)
        vectorized: np.ndarray = dilated.reshape((-1, 3))
        idxs: np.ndarray = np.array(
            [idx for idx, _ in np.ndenumerate(np.mean(dilated, axis=2))]
        )
        vectorized = np.hstack((vectorized, idxs))

        # Run Kmeans with K=2
        term_crit: tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K: int = 2
        _, label, center = cv2.kmeans(
            np.float32(vectorized),
            K=K,
            bestLabels=None,
            criteria=term_crit,
            attempts=10,
            flags=0,
        )
        center = np.uint8(center)[:, :3]

        # Convert back to BGR
        self.kmeans_img = center[label.flatten()]
        self.kmeans_img = self.kmeans_img.reshape((dilated.shape))

    def _get_color_value(self) -> np.ndarray:
        """
        Get the BGR value of the text color.

        Returns
        -------
        np.ndarray
            the color of the text
        """
        ## Find the two colors in the image ##
        img_colors: np.ndarray = np.unique(
            self.kmeans_img.reshape(-1, self.kmeans_img.shape[2]), axis=0
        )

        # Mask of Color 1
        color_1_r: np.ndarray = np.where(
            self.kmeans_img[:, :, 0] == img_colors[0][0], 1, 0
        )
        color_1_g: np.ndarray = np.where(
            self.kmeans_img[:, :, 1] == img_colors[0][1], 1, 0
        )
        color_1_b: np.ndarray = np.where(
            self.kmeans_img[:, :, 2] == img_colors[0][2], 1, 0
        )
        color_1_mat: np.ndarray = np.bitwise_and(
            color_1_r, color_1_g, color_1_b
        ).astype(np.uint8)
        color_1_adj_mat: np.ndarray = np.where(color_1_mat == 1, 255, 128).astype(
            np.uint8
        )

        # Mask of Color 2
        color_2_mat: np.ndarray = np.where(color_1_mat == 1, 0, 1).astype(np.uint8)

        ## Calculate the mean distance of colors to center ##
        # Set middle pixel to 0
        dimensions: tuple = np.shape(color_1_adj_mat)
        center_pt: tuple = (
            int(dimensions[0] / 2),
            int(dimensions[1] / 2),
        )
        color_1_adj_mat[center_pt] = 0

        # calculate distance of each pixel to center pixel
        distance_mat: np.ndarray = cv2.distanceTransform(
            color_1_adj_mat, cv2.DIST_L2, 3
        )

        # average distance for each color
        dist_1 = cv2.mean(distance_mat, color_1_mat)[0]
        dist_2 = cv2.mean(distance_mat, color_2_mat)[0]

        ## Color of text is closest to the center ##
        color: np.ndarray = (
            img_colors[0] if min(dist_1, dist_2) == dist_1 else img_colors[1]
        )

        return color

    def _parse_color(self, color_val: np.ndarray) -> str:
        """
        Parse the color value to determine the competition equivalent color.

        Parameters
        ----------
        color_val : np.ndarray
            the RGB color value of the text

        Returns
        -------
        str
            the color as a string
        """
        ## Convert color to HSV
        frame: np.ndarray = np.reshape(
            color_val, (1, 1, 3)
        )  # store as single-pixel image
        hsv_color_val: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

        ## Determine which ranges color falls in ##
        matched: list = []  # colors matched to the text
        for c, ranges in POSSIBLE_COLORS.items():
            if len(ranges) > 2:  # red has 2 ranges
                if (cv2.inRange(hsv_color_val, ranges[1], ranges[0])[0, 0] == 255) or (
                    cv2.inRange(hsv_color_val, ranges[3], ranges[2])[0, 0] == 255
                ):
                    matched.append(c)
            elif cv2.inRange(hsv_color_val, ranges[1], ranges[0])[0, 0] == 255:
                matched.append(c)

        ## Determine distance to center to choose color if falls in multiple ##
        color: str = None  # returns None if no match
        if len(matched) > 1:  # 2+ matched colors
            # find color with min dist to color value
            best_dist = 1000

            for c in matched:
                dist = 1000
                # get midpoint value of color range
                if len(POSSIBLE_COLORS[c]) > 2:  # handle red's 2 ranges
                    mid1 = np.mean(POSSIBLE_COLORS[c][:2])  # midpoint of range 1
                    mid2 = np.mean(POSSIBLE_COLORS[c][2:])  # midpoint of range 2
                    dist = min(  # min dist of color to range mid
                        np.sum(np.abs(hsv_color_val - mid1)),
                        np.sum(np.abs(hsv_color_val - mid2)),
                    )
                else:  # any color except red
                    mid = np.mean(POSSIBLE_COLORS[c])  # midpoint of range
                    dist = np.sum(
                        np.abs(hsv_color_val - mid)
                    )  # dist of color to range mid

                if dist < best_dist:  # color with min distance is the color chosen
                    best_dist = dist
                    color = c
        else:  # single matched color
            color = matched[0]

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
