"""
Functions related to detecting the text of standard objects.
"""
import cv2
import numpy as np
import pytesseract


class TextDetection:
    """
    Class for handling detection of text on standard objects.
    """

    def __init__(self):
        self.image = np.array([])

    def detect_text(self, img: np.ndarray, bounds: np.ndarray = None) -> np.ndarray:
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
        cv2.imshow("Original Image", img)  ## TODO: remove all imshows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # correct image if necessary
        corrected_img = img
        if bounds != None:
            corrected_img = self._slice_rotate_img(img, bounds)

        # Image processing to make text more clear
        processed_img = self._preprocess_img(corrected_img)

        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # output_image = np.dstack((blurred_img, blurred_img, blurred_img))
        output_image = np.dstack((processed_img, processed_img, processed_img))

        # detect text
        print("Image processing complete.")
        txt_data = pytesseract.image_to_data(
            output_image,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 10",
        )
        print(txt_data)
        # filter detected text to find valid characters
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

                        color = self._get_text_color(img, bounds)

                        # add to found characters array
                        found_characters += [(txt, bounds, color)]

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

    def _get_text_color(self, img: np.ndarray, bounds: np.ndarray) -> str:
        """
        Detect the color of the text.

        Parameters
        ----------
        img : np.ndarray
            the image the text is in
        bounds : np.ndarray
            bounds of the text

        Returns
        -------
        str
            the color of the text
        """
        ## TEMP: Initial Code to provide broad outline, out of scope of current issue
        color = "Green"

        return color

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
        # Slice image around bounds and find center point
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

        # Get angle of rotation
        tl_x = bounds[0][0]
        tr_x = bounds[3][
            0
        ]  ## TODO: 1st index depends on how bounds stored for standard object
        tl_y = bounds[0][1]
        tr_y = bounds[3][1]
        angle = np.rad2deg(np.arctan((tr_y - tl_y) / (tr_x - tl_x)))
        self.angle = angle

        # Rotate image
        rot_mat = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
        rotated_img = cv2.warpAffine(
            cropped_img, rot_mat, cropped_img.shape[1::-1], flags=cv2.INTER_LINEAR
        )

        cv2.imshow("Rotated image", rotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return rotated_img

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

        laplace_img = cv2.Laplacian(dilated, ddepth=cv2.CV_8U, ksize=5)

        # binarize image
        binarized = np.where(laplace_img > 50, np.uint8(255), np.uint8(0))

        # Additional blur to remove noise
        blur_2 = cv2.medianBlur(binarized, ksize=3)

        return blur_2


if __name__ == "__main__":
    """
    Driver for testing text detection and classification functions.
    """
    img = cv2.imread(
        "/home/cameron/Documents/GitHub/SUAS-2022/vision/Object Detection/text_y.jpg"
    )

    bounds = [[77, 184], [3, 91], [120, 0], [194, 82]]

    detector = TextDetection()

    detected_chars = detector.detect_text(img, bounds)

    print("The following characters were found in the image:", detected_chars)
