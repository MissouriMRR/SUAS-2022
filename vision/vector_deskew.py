import cv2

import coordinate_lengths
from vector_utils import *


def calculate_distance(
    pixel1: Tuple[int, int],
    pixel2: Tuple[int, int],
    image_shape: Tuple[int, int, int],
    focal_length: float,
    rotation_deg: List[float],
    altitude: float,
) -> float:
    """
    Calculates the physical distance between two points on the ground represented by pixels
    locations

    Parameters
    ----------
    pixel1, pixel2: Tuple[int, int]
        The two input pixels in [Y,X] form. The distance between them will be calculated
    image_shape : Tuple[int, int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    focal_length : float
        The camera's focal length
    rotation_deg : List[float]
        The [roll, pitch, yaw] rotation in degrees
    altitude: float
        The altitude of the drone in any units. If an altitude is given, the units of the output
        will be the units of the input.
    Returns
    -------
    distance : float
        The distance between the two pixels. Units are the same units as `altitude`
    """
    intersect1: npt.NDArray[np.float64] = pixel_intersect(
        pixel1, image_shape, focal_length, rotation_deg, altitude
    )
    intersect2: npt.NDArray[np.float64] = pixel_intersect(
        pixel2, image_shape, focal_length, rotation_deg, altitude
    )

    # Calculate the distance between the two intersects
    distance: float = np.linalg.norm(intersect1 - intersect2)

    return distance


def get_coordinates(
    pixel: Tuple[int, int],
    image_shape: Tuple[int, int, int],
    focal_length: float,
    rotation_deg: List[float],
    drone_coordinates: List[float],
    altitude_m: float,
) -> Tuple[float, float]:
    """
    Calculates the coordinates of the given pixel

    Parameters
    ----------
    pixel: Tuple[int, int]
        The coordinates of the pixel in [Y, X] form
    image_shape : Tuple[int, int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    focal_length : float
        The camera's focal length
    rotation_deg: List[float]
        The rotation of the drone/camera. The ROTATION_OFFSET in vector_utils.py will be applied
        after.
    drone_coordinates: List[float]
        The coordinates of the drone in degrees of (latitude, longitude)
    altitude_m: float
        The altitude of the drone in meters
    Returns
    -------
    pixel_coordinates : Tuple[float, float]
        The (latitude, longitude) coordinates of the pixel in degrees
    """
    # Calculate the latitude and longitude lengths (in meters)
    latitude_length = coordinate_lengths.latitude_length(drone_coordinates[0])
    longitude_length = coordinate_lengths.longitude_length(drone_coordinates[0])

    # Find the pixel's intersect with the ground to get the location relative to the drone
    intersect = pixel_intersect(pixel, image_shape, focal_length, rotation_deg, altitude_m)

    # Invert the X axis so that the longitude is correct
    intersect[1] *= -1

    # Convert the location to latitude and longitude and add it to the drone's coordinates
    pixel_lat = drone_coordinates[0] + intersect[0] / latitude_length
    pixel_lon = drone_coordinates[1] + intersect[1] / longitude_length

    return pixel_lat, pixel_lon


def deskew(
    image: npt.NDArray[np.uint8],
    focal_length: float,
    rotation_deg: List[float],
    scale: Optional[float] = 1,
    interpolation: Optional[int] = cv2.INTER_LINEAR,
) -> npt.NDArray[np.uint8]:
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
    deskewed_image : npt.NDArray[np.uint8]
        The deskewed image - the image is flattened with black areas in the margins
    """
    orig_height, orig_width, _ = image.shape

    src_pts = np.float32([[0, 0], [orig_width, 0], [orig_width, orig_height], [0, orig_height]])

    # Convert XY to YX
    flipped = np.flip(src_pts, axis=1)

    intersects = np.float32(
        [pixel_intersect(point, image.shape, focal_length, rotation_deg) for point in flipped]
    )

    # Flip the endpoints over the X axis (top left is 0,0 for images)
    intersects[:, 1] *= -1

    # Subtract the minimum on both axes so the minimum values on each axis are 0
    intersects -= intersects.min(axis=0)

    # Find the area using cv2 contour tools
    area: float = cv2.contourArea(intersects)

    # Scale the output so the area of the important pixels is about the same as the starting image
    target_area = image.shape[0] * image.shape[1] * scale
    intersect_scale = np.sqrt(target_area / area)
    dst_pts = intersects * intersect_scale

    dst_pts = np.round(dst_pts)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    result_height = int(dst_pts[:, 1].max()) + 1
    result_width = int(dst_pts[:, 0].max()) + 1

    result = cv2.warpPerspective(
        image,
        matrix,
        (result_width, result_height),
        flags=interpolation,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return result
