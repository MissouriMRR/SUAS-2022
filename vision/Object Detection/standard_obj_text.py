"""
Algorithms related to detecting the text of standard objects.
"""

from typing import Dict, List, Tuple, Optional

import cv2

import numpy as np
import numpy.typing as npt

import pytesseract

# Possible colors and HSV upper/lower bounds
POSSIBLE_COLORS: Dict[str, npt.NDArray[np.int64]] = {
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
    Characteristics are of the character, orientation, and color.
    """

    def __init__(self) -> None:
        # related to text detection
        self.rotated_img: npt.NDArray[np.uint8] = np.array([])
        self.preprocessed: npt.NDArray[np.uint8] = np.array([])

        # related to text color
        self.text_cropped_img: npt.NDArray[np.uint8] = np.array([])
        self.kmeans_img: npt.NDArray[np.uint8] = np.array([])
        self.color: Optional[str] = None

        # related to text orientation
        self.orientation: Optional[str] = None

    def get_text_characteristics(
        self, img: npt.NDArray[np.uint8], bounds: List[List[int]]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Gets the characteristics of the text on the standard object.

        Parameters
        ----------
        img : np.NDArray[np.uint8]
            image to find characteristics of text within
        bounds : List[List[int, int]]
            bounds of standard object containing text on in image

        Returns
        -------
        (character, orientation, color) : Tuple[str, str, str]
            characteristics of the text in the form
        """
        ## Get the character ##
        characters: List[Tuple[str, List[Tuple[int, int]]]] = self._detect_text(
            img, bounds
        )
        if len(characters) != 1:
            return (None, None, None)
        character: str
        char_bounds: List[Tuple[int, int]]
        character, char_bounds = characters[0]

        ## Get the orientation ##
        orientation: Optional[str] = self._get_orientation()

        ## Get the color of the text ##
        color: Optional[str] = self._get_text_color(char_bounds)

        return (character, orientation, color)

    def _detect_text(
        self, img: npt.NDArray[np.uint8], bounds: List[List[int]]
    ) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Detect text within an image.
        Will return string for parameter odlc.alphanumeric

        Parameters
        ----------
        img : npt.NDArray[np.uint8]
            image to detect text within
        bounds : List[List[int, int]]
            array of tuple bounds (4 x-y coordinates)

        Returns
        -------
        found_characters : List[Tuple[str, List[Tuple[int, int]]]]
            list containing detected characters and their bounds
        """
        ## Crop and rotate the image ##
        self._slice_rotate_img(img, bounds)

        ## Image preprocessing to make text more clear ##
        processed_img: npt.NDArray[np.uint8] = self._preprocess_img(self.rotated_img)
        output_image: npt.NDArray[np.uint8] = np.dstack(
            (processed_img, processed_img, processed_img)
        )

        ## Detect Text ##
        txt_data = pytesseract.image_to_data(
            output_image,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 10",
        )

        ## Filter detected text to find valid characters ##
        found_characters: List[Tuple[str, List[Tuple[int, int]]]] = []
        for i, txt in enumerate(txt_data["text"]):
            if (txt is not None) and (len(txt) == 1):  # length of 1
                # must be uppercase letter or number
                if (txt.isalpha() and txt.isupper()) or txt.isnumeric():
                    # get data for each text object detected
                    x = txt_data["left"][i]
                    y = txt_data["top"][i]
                    width = txt_data["width"][i]
                    height = txt_data["height"][i]

                    # Don't continue processing if text is size of full image
                    img_h, img_w = np.shape(processed_img)
                    if not (x == 0 and y == 0 and width == img_w and height == img_h):
                        t_bounds = [
                            (x, y),
                            (x + width, y),
                            (x + width, y + height),
                            (x, y + height),
                        ]

                        # add to found characters array
                        found_characters += [(txt, t_bounds)]

        return found_characters

    def _preprocess_img(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Preprocess image for text detection.

        Parameters
        ----------
        img : npt.NDArray[np.uint8]
            image to preprocess

        Returns
        -------
        blur_2 : npt.NDArray[np.uint8]
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
        self.preprocessed = cv2.medianBlur(binarized, ksize=3)

        return self.preprocessed

    def _slice_rotate_img(
        self, img: npt.NDArray[np.uint8], bounds: List[List[int]]
    ) -> None:
        """
        Slice a portion of an image and rotate to be rectangular.

        Parameters
        ----------
        img : npt.NDArray[np.uint8]
            the image to take a splice of
        bounds : List[List[int]]
            array of bound coordinates (4 x-y coordinates; tl-tr-br-bl)

        Returns
        -------
        None
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
        ## NOTE: 1st index depends on how bounds stored for standard object
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

    def _get_text_color(self, char_bounds: List[Tuple[int, int]]) -> Optional[str]:
        """
        Detect the color of the text.

        Parameters
        ----------
        char_bounds : List[Tuple[int, int]]
            bounds of the text

        Returns
        -------
        color : str
            the color of the text
        """
        # Slice rotated image around bounds of text ##
        x_vals = [coord[0] for coord in char_bounds]
        y_vals = [coord[1] for coord in char_bounds]
        min_x: int = np.amin(x_vals)
        max_x: int = np.amax(x_vals)
        min_y: int = np.amin(y_vals)
        max_y: int = np.amax(y_vals)

        self.text_cropped_img = self.rotated_img[min_y:max_y, min_x:max_x, :]

        ## Run Kmeans with K=2 ##
        self._run_kmeans()

        ## Determine which of the 2 colors is more central ##
        ## NOTE: val is in BGR due to kmeans return
        color_val: npt.NDArray[np.uint8] = self._get_color_value()

        ## Match found color to color enum ##
        color: Optional[str] = self._parse_color(color_val)

        return color

    def _run_kmeans(self) -> None:
        """
        Run kmeans with K=2 for color classification.

        Returns
        -------
        None
        """
        ## Image preprocessing to make text bigger/clearer ##
        blur: npt.NDArray[np.uint8] = cv2.medianBlur(self.text_cropped_img, ksize=9)

        kernel: npt.NDArray[np.uint8] = np.ones((5, 5), np.uint8)
        erosion: npt.NDArray[np.uint8] = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated: npt.NDArray[np.uint8] = cv2.dilate(
            erosion, kernel=kernel, iterations=1
        )

        ## Color and Location-based KMeans clustering ##

        # Convert to (R, G, B, X, Y)
        vectorized: npt.NDArray[np.uint8] = dilated.reshape((-1, 3))
        idxs: npt.NDArray[np.uint8] = np.array(
            [idx for idx, _ in np.ndenumerate(np.mean(dilated, axis=2))]
        )
        vectorized = np.hstack((vectorized, idxs))

        # Run Kmeans with K=2
        term_crit = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )
        k_val: int = 2
        _, label, center = cv2.kmeans(
            np.float32(vectorized),
            K=k_val,
            bestLabels=None,
            criteria=term_crit,
            attempts=10,
            flags=0,
        )
        center = center.astype(np.uint8)[:, :3]

        # Convert back to BGR
        self.kmeans_img = center[label.flatten()]
        self.kmeans_img = self.kmeans_img.reshape((dilated.shape))

    def _get_color_value(self) -> npt.NDArray[np.uint8]:
        """
        Get the BGR value of the text color.

        Returns
        -------
        color : npt.NDArray[np.uint8]
            the color of the text
        """
        ## Find the two colors in the image ##
        img_colors: np.ndarray = np.unique(
            self.kmeans_img.reshape(-1, self.kmeans_img.shape[2]), axis=0
        )

        # Mask of Color 1
        color_1_r: npt.NDArray[np.uint8] = np.where(
            self.kmeans_img[:, :, 0] == img_colors[0][0], 1, 0
        )
        color_1_g: npt.NDArray[np.uint8] = np.where(
            self.kmeans_img[:, :, 1] == img_colors[0][1], 1, 0
        )
        color_1_b: npt.NDArray[np.uint8] = np.where(
            self.kmeans_img[:, :, 2] == img_colors[0][2], 1, 0
        )
        color_1_mat: npt.NDArray[np.uint8] = np.bitwise_and(
            color_1_r, color_1_g, color_1_b
        ).astype(np.uint8)
        color_1_adj_mat: npt.NDArray[np.uint8] = np.where(
            color_1_mat == 1, 255, 128
        ).astype(np.uint8)

        # Mask of Color 2
        color_2_mat: npt.NDArray[np.uint8] = np.where(color_1_mat == 1, 0, 1).astype(
            np.uint8
        )

        ## Calculate the mean distance of colors to center ##
        # Set middle pixel to 0
        dimensions: Tuple[int, int] = np.shape(color_1_adj_mat)
        center_pt: Tuple[int, int] = (
            int(dimensions[0] / 2),
            int(dimensions[1] / 2),
        )
        color_1_adj_mat[center_pt] = 0

        # calculate distance of each pixel to center pixel
        distance_mat: npt.NDArray[float] = cv2.distanceTransform(
            color_1_adj_mat, cv2.DIST_L2, 3
        )

        # average distance for each color
        dist_1 = cv2.mean(distance_mat, color_1_mat)[0]
        dist_2 = cv2.mean(distance_mat, color_2_mat)[0]

        ## Color of text is closest to the center ##
        color: npt.NDArray[np.uint8] = (
            img_colors[0] if min(dist_1, dist_2) == dist_1 else img_colors[1]
        )

        return color

    def _parse_color(self, color_val: npt.NDArray[np.uint8]) -> Optional[str]:
        """
        Parse the color value to determine the competition equivalent color.

        Parameters
        ----------
        color_val : npt.NDArray[np.uint8]
            the RGB color value of the text

        Returns
        -------
        color : Optional[str]
            the color as a string
        """
        ## Convert color to HSV
        frame: npt.NDArray[np.uint8] = np.reshape(
            color_val, (1, 1, 3)
        )  # store as single-pixel image
        hsv_color_val: npt.NDArray[np.uint8] = cv2.cvtColor(
            frame, cv2.COLOR_BGR2HSV_FULL
        )

        ## Determine which ranges color falls in ##
        matched: List[str] = []  # colors matched to the text
        for col, ranges in POSSIBLE_COLORS.items():
            if len(ranges) > 2:  # red has 2 ranges
                if (cv2.inRange(hsv_color_val, ranges[1], ranges[0])[0, 0] == 255) or (
                    cv2.inRange(hsv_color_val, ranges[3], ranges[2])[0, 0] == 255
                ):
                    matched.append(col)
            elif cv2.inRange(hsv_color_val, ranges[1], ranges[0])[0, 0] == 255:
                matched.append(col)

        ## Determine distance to center to choose color if falls in multiple ##
        self.color = None  # returns None if no match
        if len(matched) > 1:  # 2+ matched colors
            # find color with min dist to color value
            best_dist = 1000

            for col in matched:
                dist = 1000
                # get midpoint value of color range
                if len(POSSIBLE_COLORS[col]) > 2:  # handle red's 2 ranges
                    mid1 = np.mean(POSSIBLE_COLORS[col][:2])  # midpoint of range 1
                    mid2 = np.mean(POSSIBLE_COLORS[col][2:])  # midpoint of range 2
                    dist = min(  # min dist of color to range mid
                        np.sum(np.abs(hsv_color_val - mid1)),
                        np.sum(np.abs(hsv_color_val - mid2)),
                    )
                else:  # any color except red
                    mid = np.mean(POSSIBLE_COLORS[col])  # midpoint of range
                    dist = np.sum(
                        np.abs(hsv_color_val - mid)
                    )  # dist of color to range mid

                if dist < best_dist:  # color with min distance is the color chosen
                    best_dist = dist
                    self.color = col
        else:  # single matched color
            self.color = matched[0]

        return self.color

    def _get_orientation(self) -> Optional[str]:
        """
        Get the orientation of the text.
        """
        ## TEMP: Implemmentation out of scope of current issue
        self.orientation = "N"

        return self.orientation


# Driver for testing text detection and classification functions.
if __name__ == "__main__":
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

    test_img: npt.NDArray[np.uint8] = cv2.imread(file_name)

    # bounds for stock image, given by standard object detection
    ## NOTE: Change to function once implemented
    test_bounds: List[List[int]] = [[77, 184], [3, 91], [120, 0], [194, 82]]

    detector = TextCharacteristics()
    detected_chars = detector.get_text_characteristics(test_img, test_bounds)

    print("The following character was found in the image:", detected_chars)
