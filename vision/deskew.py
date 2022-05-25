"""Distorts an image to generate an overhead view of the photo."""

from typing import List, Tuple, Optional

import cv2
import numpy as np
import numpy.typing as npt

from vector_utils import pixel_intersect


def deskew(
    image: npt.NDArray[np.uint8],
    focal_length: float,
    rotation_deg: List[float],
    scale: float = 1,
    interpolation: Optional[int] = cv2.INTER_LINEAR,
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64]]:
    """
    Distorts an image to generate an overhead view of the photo. Parts of the image will be
    completely black where the camera could not see.

    Parameters
    ----------
    image : npt.NDArray[np.uint8]
        The input image to deskew. Aspect ratio should match the camera sensor
    focal_length : float
        The camera's focal length - used to generate the camera's fields of view
    rotation_deg : List[float]
        The [roll, pitch, yaw] rotation in degrees
    scale: Optional[float]
        Scales the resolution of the output. A value of 1 makes the area inside the camera view
        equal to the original image. Defaults to 1.
    interpolation: Optional[int]
        The cv2 interpolation type to be used when deskewing.
    Returns
    -------
    (deskewed_image, corner_points) : Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64]]
        deskewed_image : npt.NDArray[np.uint8]
            The deskewed image - the image is flattened with black areas in the margins
        corner_points :
            The corner points of the result in the image.
            Points are in order based on their location in the original image.
            Format is: (top left, top right, bottom right, bottom left), or
            1--2
            |  |
            4--3
    """
    orig_height: int
    orig_width: int
    orig_height, orig_width, _ = image.shape

    # Generate points in the format
    # 1--2
    # |  |
    # 4--3

    src_pts: npt.NDArray[np.float32] = np.array(
        [[0, 0], [orig_width, 0], [orig_width, orig_height], [0, orig_height]], dtype=np.float32
    )
    intersects: npt.NDArray[np.float32] = np.array(
        [
            pixel_intersect(point, image.shape, focal_length, rotation_deg, 1)
            for point in np.flip(src_pts, axis=1)  # use np.flip to convert XY to YX
        ],
        dtype=np.float32,
    )

    # Flip the endpoints over the X axis (top left is 0,0 for images)
    intersects[:, 1] *= -1

    # Subtract the minimum on both axes so the minimum values on each axis are 0
    intersects -= np.min(intersects, axis=0)

    # Find the area using cv2 contour tools
    area: float = cv2.contourArea(intersects)

    # Scale the output so the area of the important pixels is about the same as the starting image
    target_area: float = float(image.shape[0]) * (float(image.shape[1]) * scale)
    intersect_scale: np.float64 = np.float64(np.sqrt(target_area / area))
    dst_pts: npt.NDArray[np.float64] = intersects * intersect_scale

    dst_pts = np.round(dst_pts)

    matrix: npt.NDArray[np.float64] = cv2.getPerspectiveTransform(src_pts, dst_pts)

    result_height: int = int(np.max(dst_pts[:, 1])) + 1
    result_width: int = int(np.max(dst_pts[:, 0])) + 1

    result: npt.NDArray[np.uint8] = cv2.warpPerspective(
        image,
        matrix,
        (result_width, result_height),
        flags=interpolation,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return result, dst_pts.astype(np.int32)
