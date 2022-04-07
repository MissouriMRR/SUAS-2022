"""
Algorithms related to detecting the text of standard objects.
"""

from typing import Dict, List, Tuple, Optional

import cv2

import numpy as np
import numpy.typing as npt

import pytesseract

from vision.common.bounding_box import ObjectType, BoundingBox


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

    def __init__(self, img: Optional[npt.NDArray[np.uint8]] = None) -> None:
        # Image to operate on
        self._img: npt.NDArray[np.uint8] = img if img is not None else np.array([])

        # related to text detection
        self._rotated_img: npt.NDArray[np.uint8] = np.array([])
        self._preprocessed: npt.NDArray[np.uint8] = np.array([])

        # related to text color
        self._text_cropped_img: npt.NDArray[np.uint8] = np.array([])
        self._kmeans_img: npt.NDArray[np.uint8] = np.array([])
        self._color: Optional[str] = None

        # related to text orientation
        # NOTE: not implemented
        # self.orientation: Optional[str] = None

    @property
    def img(self) -> npt.NDArray[np.uint8]:
        """
        Getter for _img. Gets the current image being processed.

        Returns
        -------
        _img : npt.NDArray[np.uint8]
            The image to find characteristics in.
        """
        return self._img

    @img.setter
    def img(self, image: npt.NDArray[np.uint8]) -> None:
        """
        Setter for _img. Sets the image to process.

        Parameters
        ----------
        image: npt.NDArray[np.uint8]
            The image to process.
        """
        self._img = image

    def get_text_characteristics(
        self, bounds: BoundingBox
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Gets the characteristics of the text on the standard object.

        Parameters
        ----------
        bounds : BoundingBox
            bounds of standard object containing text on in image

        Returns
        -------
        (character, orientation, color) : Tuple[str | None, str | None, str | None]
            characteristics of the text in the form.
            Output is None when characteristic was not found, either because
            the associated function failed or because text detection failed.
        """
        ## Get the character ##
        characters: List[Tuple[str, BoundingBox]] = self.detect_text(bounds)
        if len(characters) != 1:
            return (None, None, None)
        character: str
        char_bounds: BoundingBox
        character, char_bounds = characters[0]

        ## Get the orientation ##
        # NOTE: Not implemented
        # orientation: Optional[str] = self._get_orientation()
        orientation: Optional[str] = "N"

        ## Get the color of the text ##
        color: Optional[str] = self.get_text_color(char_bounds)

        return (character, orientation, color)

    def detect_text(self, bounds: BoundingBox) -> List[Tuple[str, BoundingBox]]:
        """
        Detect text within an image.
        Will return string for parameter odlc.alphanumeric

        Parameters
        ----------
        bounds : BoundingBox
            BoundingBox of standard object on which to detect text.

        Returns
        -------
        found_characters : List[Tuple[str, BoundingBox]]
            list containing detected characters and their bounds
        """
        ## Crop and rotate the image ##
        self._slice_rotate_img(self._img, bounds)

        ## Image preprocessing to make text more clear ##
        processed_img: npt.NDArray[np.uint8] = self._preprocess_img(self._rotated_img)
        output_image: npt.NDArray[np.uint8] = np.dstack(
            (processed_img, processed_img, processed_img)
        )

        ## Detect Text ##
        txt_data: Dict[str, list] = pytesseract.image_to_data(
            output_image,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 10",
        )

        ## Filter detected text to find valid characters ##
        found_characters: List[Tuple[str, BoundingBox]] = []
        for i, txt in enumerate(txt_data["text"]):
            if (txt is not None) and (len(txt) == 1):  # length of 1
                # must be uppercase letter or number
                if (txt.isalpha() and txt.isupper()) or txt.isnumeric():
                    # get data for each text object detected
                    x: int = txt_data["left"][i]
                    y: int = txt_data["top"][i]
                    width: int = txt_data["width"][i]
                    height: int = txt_data["height"][i]

                    # Don't continue processing if text is size of full image
                    img_h: int
                    img_w: int
                    img_h, img_w = np.shape(processed_img)
                    if not (x == 0 and y == 0 and width == img_w and height == img_h):
                        t_bounds: Tuple[
                            Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]
                        ] = (
                            (x, y),
                            (x + width, y),
                            (x + width, y + height),
                            (x, y + height),
                        )
                        text_bb = BoundingBox(t_bounds, ObjectType.TEXT)

                        # add to found characters array
                        found_characters += [(txt, text_bb)]

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
        gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # blur to remove noise
        blur: npt.NDArray[np.uint8] = cv2.medianBlur(gray, ksize=9)

        # erode and dilate to increase text clarity and reduce noise
        kernel: npt.NDArray[np.uint8] = np.ones((5, 5), np.uint8)
        eroded: npt.NDArray[np.uint8] = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated: npt.NDArray[np.uint8] = cv2.dilate(eroded, kernel=kernel, iterations=1)

        # laplace edge detection
        laplace_img: npt.NDArray[np.uint8] = cv2.Laplacian(dilated, ddepth=cv2.CV_8U, ksize=5)

        # binarize image
        binarized: npt.NDArray[np.uint8] = np.where(laplace_img > 50, np.uint8(255), np.uint8(0))

        # additional blur to remove noise
        self._preprocessed = cv2.medianBlur(binarized, ksize=3)

        return self._preprocessed

    def _slice_rotate_img(self, img: npt.NDArray[np.uint8], bounds: BoundingBox) -> None:
        """
        Slice a portion of an image and rotate to be rectangular.

        Parameters
        ----------
        img : npt.NDArray[np.uint8]
            the image to take a splice of
        bounds : BoundingBox
            array of bound coordinates (4 x-y coordinates; tl-tr-br-bl)
        """
        ## Slice image around bounds and find center point ##
        min_x: int
        max_x: int
        min_x, max_x = bounds.get_x_extremes()

        min_y: int
        max_y: int
        min_y, max_y = bounds.get_y_extremes()

        cropped_img: npt.NDArray[np.uint8] = img[min_y:max_y, min_x:max_x, :]

        dimensions: Tuple[int, int, int] = np.shape(cropped_img)
        center_pt: Tuple[int, int] = (
            int(dimensions[0] / 2),
            int(dimensions[1] / 2),
        )

        ## Get angle of rotation ##
        angle: float = bounds.get_rotation_angle()

        ## Rotate image ##
        rot_mat: npt.NDArray[np.uint8] = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
        self._rotated_img = cv2.warpAffine(
            cropped_img, rot_mat, cropped_img.shape[1::-1], flags=cv2.INTER_LINEAR
        )

    def get_text_color(self, char_bounds: BoundingBox) -> Optional[str]:
        """
        Detect the color of the text.

        Parameters
        ----------
        char_bounds : BoundingBox
            BoundingBox around the text

        Returns
        -------
        color : Optional[str]
            The color of the text. Possible colors are the string keys of POSSIBLE_COLORS.
            Returns None if HSV color value did not fall in a range defined in POSSIBLE_COLORS.
        """
        # Slice rotated image around bounds of text ##
        min_x: int
        max_x: int
        min_x, max_x = char_bounds.get_x_extremes()

        min_y: int
        max_y: int
        min_y, max_y = char_bounds.get_y_extremes()

        self._text_cropped_img = self._rotated_img[min_y:max_y, min_x:max_x, :]

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
        """
        ## Image preprocessing to make text bigger/clearer ##
        blur: npt.NDArray[np.uint8] = cv2.medianBlur(self._text_cropped_img, ksize=9)

        kernel: npt.NDArray[np.uint8] = np.ones((5, 5), np.uint8)
        erosion: npt.NDArray[np.uint8] = cv2.erode(blur, kernel=kernel, iterations=1)
        dilated: npt.NDArray[np.uint8] = cv2.dilate(erosion, kernel=kernel, iterations=1)

        ## Color and Location-based KMeans clustering ##

        # Convert to (R, G, B, X, Y)
        vectorized: npt.NDArray[np.uint8] = dilated.reshape((-1, 3))
        idxs: npt.NDArray[np.uint8] = np.array(
            [idx for idx, _ in np.ndenumerate(np.mean(dilated, axis=2))]
        )
        vectorized = np.hstack((vectorized, idxs))

        # Run Kmeans with K=2
        term_crit: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )
        k_val: int = 2

        label: npt.NDArray[str]
        center: npt.NDArray[np.float32]
        _, label, center = cv2.kmeans(
            np.float32(vectorized),
            K=k_val,
            bestLabels=None,
            criteria=term_crit,
            attempts=10,
            flags=0,
        )
        center_int: npt.NDArray[np.uint8] = center.astype(np.uint8)[:, :3]

        # Convert back to BGR
        self._kmeans_img = center_int[label.flatten()]
        self._kmeans_img = self._kmeans_img.reshape((dilated.shape))

    def _get_color_value(self) -> npt.NDArray[np.uint8]:
        """
        Get the BGR value of the text color.

        Returns
        -------
        color : npt.NDArray[np.uint8]
            the color of the text
        """
        ## Find the two colors in the image ##
        img_colors: npt.NDArray[np.uint8] = np.unique(
            self._kmeans_img.reshape(-1, self._kmeans_img.shape[2]), axis=0
        )

        # Mask of Color 1
        color_1_r: npt.NDArray[np.uint8] = np.where(
            self._kmeans_img[:, :, 0] == img_colors[0][0], 1, 0
        )
        color_1_g: npt.NDArray[np.uint8] = np.where(
            self._kmeans_img[:, :, 1] == img_colors[0][1], 1, 0
        )
        color_1_b: npt.NDArray[np.uint8] = np.where(
            self._kmeans_img[:, :, 2] == img_colors[0][2], 1, 0
        )
        color_1_mat: npt.NDArray[np.uint8] = np.bitwise_and(color_1_r, color_1_g, color_1_b).astype(
            np.uint8
        )
        color_1_adj_mat: npt.NDArray[np.uint8] = np.where(color_1_mat == 1, 255, 128).astype(
            np.uint8
        )

        # Mask of Color 2
        color_2_mat: npt.NDArray[np.uint8] = np.where(color_1_mat == 1, 0, 1).astype(np.uint8)

        ## Calculate the mean distance of colors to center ##
        # Set middle pixel to 0
        dimensions: Tuple[int, int] = np.shape(color_1_adj_mat)
        center_pt: Tuple[int, int] = (
            int(dimensions[0] / 2),
            int(dimensions[1] / 2),
        )
        color_1_adj_mat[center_pt] = 0

        # calculate distance of each pixel to center pixel
        distance_mat: npt.NDArray[float] = cv2.distanceTransform(color_1_adj_mat, cv2.DIST_L2, 3)

        # average distance for each color
        dist_1: float = cv2.mean(distance_mat, color_1_mat)[0]
        dist_2: float = cv2.mean(distance_mat, color_2_mat)[0]

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
        hsv_color_val: npt.NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

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
        self._color = None  # returns None if no match
        if len(matched) > 1:  # 2+ matched colors
            # find color with min dist to color value
            best_dist: float = float("inf")

            for col in matched:
                dist: float = float("inf")
                # get midpoint value of color range
                if len(POSSIBLE_COLORS[col]) > 2:  # handle red's 2 ranges
                    mid1: float = np.mean(POSSIBLE_COLORS[col][:2])  # midpoint of range 1
                    mid2: float = np.mean(POSSIBLE_COLORS[col][2:])  # midpoint of range 2
                    dist = min(  # min dist of color to range mid
                        np.sum(np.abs(hsv_color_val - mid1)),
                        np.sum(np.abs(hsv_color_val - mid2)),
                    )
                else:  # any color except red
                    mid: float = np.mean(POSSIBLE_COLORS[col])  # midpoint of range
                    dist = np.sum(np.abs(hsv_color_val - mid))  # dist of color to range mid

                if dist < best_dist:  # color with min distance is the color chosen
                    best_dist = dist
                    self._color = col
        else:  # single matched color
            self._color = matched[0]

        return self._color

    def get_orientation(self) -> Optional[str]:
        """
        Get the orientation of the text.
        """
        raise NotImplementedError("Orientation not implemented.")


# Driver for testing text detection and classification functions.
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Runs text characteristics algorithms. Must specify a file."
    )

    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="Filename of the image. Required argument.",
    )

    args: argparse.Namespace = parser.parse_args()

    # no benchmark name specified, cannot continue
    if not args.file_name:
        raise RuntimeError("No file specified.")
    file_name: str = args.file_name

    test_img: npt.NDArray[np.uint8] = cv2.imread(file_name)

    # bounds for stock image, given by standard object detection
    ## NOTE: Change to function once implemented
    test_bounds: BoundingBox = BoundingBox(
        ((77, 184), (3, 91), (120, 0), (194, 82)), ObjectType.STD_OBJECT
    )

    detector: TextCharacteristics = TextCharacteristics()
    detector.img = test_img
    detected_chars: Tuple[
        Optional[str], Optional[str], Optional[str]
    ] = detector.get_text_characteristics(test_bounds)

    print("The following character was found in the image:", detected_chars)
