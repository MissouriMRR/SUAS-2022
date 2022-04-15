"""
Algorithms related to stitching images.
"""

import os
import argparse
from typing import List, Tuple
import numpy.typing as npt
import numpy as np
import cv2


class Stitcher:
    """
    Class for stitching images together and formatting them into competition standards.
    Standards include: no black borders, 16:9 aspect ratio, and WGS84 projection.
    """

    def __init__(self) -> None:
        self.final_image: npt.NDArray[np.uint8] = np.array([])
        self.color_images: List[npt.NDArray[np.uint8]] = []
        self.image_path: str = ""
        self.matches: npt.NDArray[np.float64] = np.array([])
        self.center_image: npt.NDArray[np.uint8] = np.array([])
        self.black_pixels: int = 0

    def multiple_image_stitch(self) -> npt.NDArray[np.uint8]:
        """
        Takes images from the filepath and runs them through stitching algorithms.
        Returns
        -------
        npt.NDArray[np.uint8]
            Final stitched image
        """
        # Make a list of all images in directory
        list_paths: List[str] = []
        for file in os.listdir(self.image_path):
            if file.endswith(".JPG") or file.endswith(".jpg"):
                list_paths.append(os.path.join(self.image_path, file))

        if len(list_paths) == 0:
            raise IndexError("Not Enough Images in Directory")

        # Read the images and appends them into a list and count black pixels
        for img in list_paths:
            color_img: np.ndarray = cv2.imread(img)
            self.color_images.append(color_img)
            self.black_pixels += (color_img == (0, 0, 0)).all(axis=-1).sum()

        # Set the first image as final_image so it can run in a loop
        self.final_image = self.color_images[0]

        # Loop through all images in images list
        for i in range(1, len(self.color_images)):
            self.get_matches(self.final_image, self.color_images[i])
            self.warp_images(self.color_images[i], self.final_image, self.matches)

            ## Debug Code: Shows each iteration and which iteration stitcher is on
            # print("Iteration:", i)
            # cv2.imshow("WIP Final", self.final_image)
            # cv2.waitKey(0)

        # Crop final image of black space
        self.crop_space(self.final_image)

        return self.final_image

    def get_matches(self, img_1: npt.NDArray[np.uint8], img_2: npt.NDArray[np.uint8]) -> None:
        """
        Finds matches between two grey images and establish a homography graph to warp two images.
        Parameters
        ----------
        img_1 : npt.NDArray[np.uint8]
            first image
        img_2 : npt.NDArray[np.uint8]
            second image
        """

        # Get a greyscale version of images
        grey_img_1: np.ndarray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        grey_img_2: np.ndarray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        # Create ORB detector to detect keypoints and descriptors
        orb: cv2.ORB = cv2.ORB_create(nfeatures=2000)

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(grey_img_1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(grey_img_2, None)

        # Create a BFMatcher object to find matching keypoints.
        bf_match: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # Find matching points
        matches: Tuple[cv2.DMatch] = bf_match.knnMatch(descriptors1, descriptors2, k=2)

        # Put matches in list
        all_matches: List[cv2.DMatch] = []
        for match_0, match_1 in matches:
            all_matches.append(match_0)

        # Finding the best matches
        good: List[cv2.Dmatch] = []
        for match_0, match_1 in matches:
            if match_0.distance < 0.6 * match_1.distance:
                good.append(match_0)

        # Set minimum match condition
        min_match_count: int = 10

        if len(good) > min_match_count:
            # Convert keypoints to an argument for findHomography
            src_pts: npt.NDArray[np.float32] = np.array(
                [keypoints1[m.queryIdx].pt for m in good], dtype=np.float32
            ).reshape((-1, 1, 2))
            dst_pts: npt.NDArray[np.float32] = np.array(
                [keypoints2[m.trainIdx].pt for m in good], dtype=np.float32
            ).reshape((-1, 1, 2))

            # Establish a homography
            matches, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        else:
            raise cv2.error("Not Enough Matches")

        self.matches = matches

    def warp_images(
        self,
        img_1: npt.NDArray[np.uint8],
        img_2: npt.NDArray[np.uint8],
        map_0: npt.NDArray[np.float64],
    ) -> None:
        """
        Warps the perspective of the images based on the homography map and
        stitches the images together based off of the points.
        ----------
        img_1 : npt.NDArray[np.uint8]
            first image
        img_2 : npt.NDArray[np.uint8]
            second image
        map_0 : npt.NDArray[np.float64]
            Homography map with relation points
        """
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

        x_min: npt.NDArray[np.int32] = np.array(
            (list_of_points.min(axis=0).ravel() - 0.5)[0], dtype=np.int32
        )
        y_min: npt.NDArray[np.int32] = np.array(
            (list_of_points.min(axis=0).ravel() - 0.5)[1], dtype=np.int32
        )
        x_max: npt.NDArray[np.int32] = np.array(
            (list_of_points.max(axis=0).ravel() + 0.5)[0], dtype=np.int32
        )
        y_max: npt.NDArray[np.int32] = np.array(
            (list_of_points.max(axis=0).ravel() + 0.5)[1], dtype=np.int32
        )

        translation_dist: List[np.int32] = [-x_min, -y_min]

        h_translation: npt.NDArray[np.int32] = np.array(
            [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
        )

        # Warp second images based off of points
        output_img: npt.NDArray[np.uint8] = cv2.warpPerspective(
            img_2, h_translation.dot(map_0), (x_max - x_min, y_max - y_min)
        )
        output_img[
            translation_dist[1] : rows1 + translation_dist[1],
            translation_dist[0] : cols1 + translation_dist[0],
        ] = img_1

        self.final_image = output_img

    def crop_space(self, img: npt.NDArray[np.uint8]) -> None:
        """
        Crops out all of the black space created from the perspective
        shift when stitching the image.
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
        while cv2.countNonZero(sub) > self.black_pixels:
            min_rect = cv2.erode(min_rect, None, iterations=10)
            sub = cv2.subtract(min_rect, thresh)

            ## Debug Code: Shows the crop and prints the number of black pixels
            # cv2.imshow("TEST", sub)
            # cv2.waitKey(0)
            # print(cv2.countNonZero(sub))

        # Finds the contours in the mask and extracts the bounding box coords
        cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        con = max(cnts, key=cv2.contourArea)
        (x, y, width, height) = cv2.boundingRect(con)

        self.final_image = stitched[y : y + height, x : x + width]

    def crop_ratio(self) -> None:
        """
        Crops final image into a 16:9 aspect ratio given the center.
        NOTE: Not yet implemeneted.
        """
        raise NotImplementedError("Function not Implemented")

    def wgs_transform(self) -> None:
        """
        Transform final image to WGS84.
        NOTE: Not yet implemeneted.
        """
        raise NotImplementedError("Function not Implemented")


if __name__ == "__main__":
    stitch: Stitcher = Stitcher()

    # Add directory of folder with images
    parser = argparse.ArgumentParser(description="Directory of images needed to be stitched.")
    parser.add_argument("-i", "--images", type=str, required=True, help="Directory path to images.")
    args: argparse.Namespace = parser.parse_args()

    stitch.image_path = args.images
    final: npt.NDArray[np.uint8] = stitch.multiple_image_stitch()

    SCALE_PERCENT: int = 100  # percent of original size
    wid: int = int(final.shape[1] * SCALE_PERCENT / 100)
    hei: int = int(final.shape[0] * SCALE_PERCENT / 100)
    dim: Tuple[int, int] = (wid, hei)

    final = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Final.jpg", final)
    cv2.waitKey(0)
    cv2.imwrite("Final.jpeg", final)
