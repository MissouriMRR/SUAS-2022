"""Functions for calculating locations of objects in an image"""

from typing import Tuple, List

import numpy as np
import numpy.typing as npt

from vision.common import coordinate_lengths
from vision.common.vector_utils import pixel_intersect


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
    latitude_length: float = coordinate_lengths.latitude_length(drone_coordinates[0])
    longitude_length: float = coordinate_lengths.longitude_length(drone_coordinates[0])

    # Find the pixel's intersect with the ground to get the location relative to the drone
    intersect: npt.NDArray[np.float64] = pixel_intersect(
        pixel, image_shape, focal_length, rotation_deg, altitude_m
    )

    # Invert the X axis so that the longitude is correct
    intersect[1] *= -1

    # Convert the location to latitude and longitude and add it to the drone's coordinates
    pixel_lat = drone_coordinates[0] + intersect[0] / latitude_length
    pixel_lon = drone_coordinates[1] + intersect[1] / longitude_length

    return pixel_lat, pixel_lon


def calculate_distance(
    pixel1: Tuple[int, int],
    pixel2: Tuple[int, int],
    image_shape: Tuple[int, int, int],
    focal_length: float,
    rotation_deg: List[float],
    altitude: float,
) -> float:
    """
    Calculates the physical distance between two points on the ground represented by pixel
    locations. Units of `distance` are the same as the units of `altitude`

    Parameters
    ----------
    pixel1, pixel2: Tuple[int, int]
        The two input pixel locations in [Y,X] form. The distance between them will be calculated
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
    distance: float = float(np.linalg.norm(intersect1 - intersect2))

    return distance
