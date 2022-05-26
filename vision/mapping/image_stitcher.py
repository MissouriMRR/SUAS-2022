"""
Algorithms related to stitching images.
"""

import os
import argparse
from typing import List, Tuple
import numpy.typing as npt
import numpy as np
import cv2
from vision.common import 
import pyproj  # pylint: disable=W0611


class Stitcher:
    """
    Class for stitching images together and formatting them into competition standards.
    Standards include: no black borders, 16:9 aspect ratio, and WGS84 projection, and
    template matching.

    Parameters
    ----------
    image_path: str
        Folder path for images needed to be stitched
    center_image: npt.NDArray[np.uint8]
        Image that should be the center of the competition area
    target_pixel: tuple
        Pixel location of desired center coordinate of final stitched image
    """

    def __init__(self) -> None:
        self.image_path: str = ""

        self.center_image: npt.NDArray[np.uint8] = np.array([])

        self.target_pixel: tuple = ()

    def multiple_image_stitch(self) -> npt.NDArray[np.uint8]:
        """
        Takes images from the filepath and runs them through stitching algorithms.

        Returns
        -------
        npt.NDArray[np.uint8]
            Final stitched image

        Raises
        ------
        IndexError
            Flags if there are no images in directory.
        """

        if not os.listdir(self.image_path):
            raise IndexError("Not Enough Images in Directory")

        # Make a list of all images in directory and counts black pixels
        color_images: List[npt.NDArray[np.uint8]] = []
        black_pixels: int = 0
        for file in os.listdir(self.image_path):
            if file.endswith(".JPG") or file.endswith(".jpg"):
                c_img: npt.NDArray[np.uint8] = cv2.imread(os.path.join(self.image_path, file))
                color_images.append(c_img)
                blk_range = cv2.inRange(c_img, (0, 0, 0), (0, 0, 0))
                black_pixels += cv2.countNonZero(blk_range)

        # Set the first image as final_image so it can run in a loop
        final_image: npt.NDArray[np.uint8] = color_images[0]

        # Loop through all images in images list
        for img in color_images[1:]:
            matches: npt.NDArray[np.float64] = self.get_matches(final_image, img)
            final_image = self.warp_images(img, final_image, matches)

            ## Debug Code: Shows each iteration and which iteration stitcher is on # pylint: disable=fixme
            # test = resize(final_image, 30)
            # cv2.imshow("WIP Final", test)
            # cv2.waitKey(0)

        # Crop final image of black space
        final_image = self.crop_space(final_image, black_pixels)

        # Crop final image to match desired center coordinate with image center
        final_image = self.template_match(final_image)

        return final_image

    @classmethod
    def get_matches(
        cls, img_1: npt.NDArray[np.uint8], img_2: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float64]:
        """
        Finds matches between two grey images and establish a homography graph to warp two images.

        Parameters
        ----------
        img_1 : npt.NDArray[np.uint8]
            first image
        img_2 : npt.NDArray[np.uint8]
            second image

        Returns
        -------
        npt.NDArray[np.float64]
            Matches between two images

        Raises
        ------
        cv2.error
            Flags if there is not enough matches between images.
        """

        # Get a greyscale version of images
        grey_img_1: npt.NDArray[np.uint8] = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        grey_img_2: npt.NDArray[np.uint8] = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        # Create ORB detector to detect keypoints and descriptors
        orb: cv2.ORB = cv2.ORB_create(nfeatures=2000)

        # Find the key points and descriptors with ORB
        keypoints1: cv2.KeyPoint
        keypoints2: cv2.KeyPoint
        descriptors1: npt.NDArray[np.uint8]
        descriptors2: npt.NDArray[np.uint8]

        keypoints1, descriptors1 = orb.detectAndCompute(grey_img_1, mask=None)
        keypoints2, descriptors2 = orb.detectAndCompute(grey_img_2, mask=None)

        # Create a BFMatcher object to find matching keypoints.
        bf_match: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # Find matching points
        matches: Tuple[cv2.DMatch] = bf_match.knnMatch(descriptors1, descriptors2, k=2)

        # Finding the best matches
        best_matches: List[cv2.DMatch] = [
            match_0 for match_0, match_1 in matches if match_0.distance < 0.6 * match_1.distance
        ]

        # Set minimum match condition
        min_match_count: int = 10

        if len(best_matches) > min_match_count:
            # Convert keypoints to an argument for findHomography
            src_pts: npt.NDArray[np.float32] = np.array(
                [keypoints1[m.queryIdx].pt for m in best_matches], dtype=np.float32
            ).reshape((-1, 1, 2))
            dst_pts: npt.NDArray[np.float32] = np.array(
                [keypoints2[m.trainIdx].pt for m in best_matches], dtype=np.float32
            ).reshape((-1, 1, 2))

            # Establish a homography
            matches_arr: npt.NDArray[np.float64]
            matches_arr, _ = cv2.findHomography(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )

        else:
            raise cv2.error("Not Enough Matches")

        return matches_arr

    @classmethod
    def warp_images(
        cls,
        img_1: npt.NDArray[np.uint8],
        img_2: npt.NDArray[np.uint8],
        map_0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        """
        Warps the perspective of the images based on the homography map and
        stitches the images together based off of the points.

        Parameters
        ----------
        img_1 : npt.NDArray[np.uint8]
            first image
        img_2 : npt.NDArray[np.uint8]
            second image
        map_0 : npt.NDArray[np.float64]
            Homography map with relation points

        Returns
        -------
        npt.NDArray[np.uint8]
            Stitched image
        """

        rows1: int
        cols1: int
        rows2: int
        cols2: int

        rows1, cols1 = img_1.shape[:2]
        rows2, cols2 = img_2.shape[:2]

        # Create list of coordinates from reference image and second image
        list_of_points_1: npt.NDArray[np.float32] = np.array(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], dtype=np.float32
        ).reshape((-1, 1, 2))
        temp_points: npt.NDArray[np.float32] = np.array(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]], dtype=np.float32
        ).reshape((-1, 1, 2))

        # Changes the field of view of the second image to the same as homography
        list_of_points_2: npt.NDArray[np.float32] = cv2.perspectiveTransform(temp_points, map_0)

        list_of_points: npt.NDArray[np.float32] = np.concatenate(
            (list_of_points_1, list_of_points_2), axis=0
        )

        m_points_max: npt.NDArray[np.float32] = list_of_points.max(axis=0).ravel()
        m_points_min: npt.NDArray[np.float32] = list_of_points.min(axis=0).ravel()

        x_min: int = int((m_points_min - 0.5)[0])
        y_min: int = int((m_points_min - 0.5)[1])
        x_max: int = int((m_points_max + 0.5)[0])
        y_max: int = int((m_points_max + 0.5)[1])

        h_translation: npt.NDArray[np.int32] = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Warp second images based off of points
        output_img: npt.NDArray[np.uint8] = cv2.warpPerspective(
            img_2, h_translation.dot(map_0), (x_max - x_min, y_max - y_min)
        )
        output_img[
            -y_min : rows1 + -y_min,
            -x_min : cols1 + -x_min,
        ] = img_1

        return output_img

    @classmethod
    def crop_space(cls, img: npt.NDArray[np.uint8], black_pixels: int) -> npt.NDArray[np.uint8]:
        """
        Crops out all of the black space created from the perspective
        shift when stitching the image.

        Parameters
        ----------
        img: npt.NDArray[np.uint8]
            Final image to be cropped.
        black_pixels: int
            Max number of black pixels in final image.

        Returns
        -------
        npt.NDArray[np.uint8]
            Cropped image
        """

        # Creates a 10 pixel border for the stitched image to help find contours
        stitched: npt.NDArray[np.uint8] = cv2.copyMakeBorder(
            img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
        )

        # Creates a grayscale version of stitched image
        gray: npt.NDArray[np.uint8] = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh: npt.NDArray[np.uint8] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds the greatest contour of image
        cnts: npt.NDArray[np.intc] = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0]
        con: npt.NDArray[np.intc] = max(cnts, key=cv2.contourArea)

        # Creates a mask of the same size as optimal image
        mask: npt.NDArray[np.uint8] = np.zeros(thresh.shape, dtype="uint8")
        (x, y, width, height) = cv2.boundingRect(con)
        cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)

        # Creates copies of our mask
        min_rect: npt.NDArray[np.uint8] = mask.copy()
        sub: npt.NDArray[np.uint8] = mask.copy()

        # Will loop until there are no more non zero pixels
        while cv2.countNonZero(sub) > black_pixels:
            min_rect = cv2.erode(min_rect, None, iterations=50)
            sub = cv2.subtract(min_rect, thresh)

            ## Debug Code: Shows the crop and prints the number of black pixels # pylint: disable=fixme
            # test = resize(sub, 70)
            # cv2.imshow("TEST", test)
            # cv2.waitKey(0)
            # print(cv2.countNonZero(sub))

        # Finds the contours in the mask and extracts the bounding box coords
        cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        con = max(cnts, key=cv2.contourArea)
        (x, y, width, height) = cv2.boundingRect(con)

        return stitched[y : y + height, x : x + width]

    def template_match(self, simg: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Uses the center image of the stitched image to calculate
        how far the current center of the stitched image is from the desired center
        of the stitched image and then crops the final stitched image
        to make the centers align.

        Parameters
        ----------
        simg: npt.NDArray[np.uint8]
            Final stitched image

        Returns
        -------
        npt.NDArray[np.uint8]
            Cropped version of stitched image
        """

        ch, cw = self.center_image.shape[:2]
        sh, sw = simg.shape[:2]
        cpix: tuple = (int(ch / 2), int(cw / 2))
        dfxy: tuple = (cpix[0] - int(self.target_pixel[0]), cpix[1] - int(self.target_pixel[1]))

        if dfxy[0] < 0:
            if dfxy[1] < 0:
                cropImg: npt.NDArray[np.uint8] = simg[abs(dfxy[0] * 2):sh, abs(dfxy[1] * 2):sw]
            else:
                cropImg: npt.NDArray[np.uint8] = simg[abs(dfxy[0] * 2):sh, 0:sw - (dfxy[1] * 2)]
        elif dfxy[0] > 0:
            if dfxy[1] < 0:
                cropImg: npt.NDArray[np.uint8] = simg[0:sh - (dfxy[0] * 2), abs(dfxy[1] * 2):sw]
            else:
                cropImg: npt.NDArray[np.uint8] = simg[0:sh - (dfxy[0] * 2), 0:sw - (dfxy[1] * 2)]

        return cropImg

    def crop_ratio(self, simg: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Crops final image into a 16:9 aspect ratio given the center.
        NOTE: Not yet implemeneted.

        Parameters
        ----------
        simg: npt.NDArray[np.uint8]
            Final stitched image

        Returns
        -------
        npt.NDArray[np.uint8]
            A 16:9 version of the final stitched image with the correct center
        """






    def wgs_transform(self) -> None:
        """
        Transform final image to WGS84.
        NOTE: Not yet implemeneted.
        """

        raise NotImplementedError("Function not Implemented")


def resize(img: npt.NDArray[np.uint8], scale: int) -> npt.NDArray[np.uint8]:
    """
    Resizes Image to the scale amount of original image.

    Parameters
    ----------
    img: npt.NDArray[np.uint8]
        Final image to be cropped.
    scale: int
        The amount of remaining image.

    Returns
    -------
    npt.NDArray[np.uint8]
        resized image.
    """
    scale_percent: int = scale  # percent of original size
    wid: int = int(img.shape[1] * scale_percent / 100)
    hei: int = int(img.shape[0] * scale_percent / 100)
    dim: Tuple[int, int] = (wid, hei)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    stitch: Stitcher = Stitcher()

    # Add directory of folder with images
    parser = argparse.ArgumentParser(description="Directory of images needed to be stitched.")
    parser.add_argument("-i", "--images", type=str, required=True, help="Directory path to images.")
    args: argparse.Namespace = parser.parse_args()

    stitch.image_path = args.images
    final = stitch.multiple_image_stitch()

    final = resize(final, 30)

    cv2.imshow("Final.jpg", final)
    cv2.waitKey(0)
    cv2.imwrite("Final.jpeg", final)
