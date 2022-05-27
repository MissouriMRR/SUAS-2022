"""Functions that use vectors to calculate camera intersections with the ground"""

from typing import List, Tuple
import numpy.typing as npt

import numpy as np
from scipy.spatial.transform import Rotation

# Sony RX100 VII sensor size
SENSOR_WIDTH = 13.2
SENSOR_HEIGHT = 8.8

# The rotation offset of the camera to the drone. The offset is applied in pixel_intersect
# Set to [0.0, -90.0, 0.0] when the camera is facing directly downwards
ROTATION_OFFSET = [0.0, 0.0, 0.0]


def pixel_intersect(
    pixel: Tuple[int, int],
    image_shape: Tuple[int, ...],
    focal_length: float,
    rotation_deg: List[float],
    height: float,
) -> npt.NDArray[np.float64]:
    """
    Finds the intersection [X,Y] of a given pixel with the ground relative to the camera.
    A camera with no rotation points in the +X direction and is centered at [0, 0, height].

    Parameters
    ----------
    pixel : Tuple[int, int]
        The coordinates of the pixel in [Y, X] form
    image_shape : Tuple[int, int, int]
        The shape of the image (returned by image.shape when image is a numpy image array)
    focal_length : float
        The camera's focal length
    rotation_deg : List[float]
        The [roll, pitch, yaw] rotation in degrees
    height : float
        The height that the image was taken at. The units of the output will be the units of the
        input.
    Returns
    -------
    intersect : npt.NDArray[np.float64]
        The coordinates [X,Y] where the pixel's vector intersects with the ground.
    """

    # Create the normalized vector representing the direction of the given pixel
    vector: npt.NDArray[np.float64] = pixel_vector(pixel, image_shape, focal_length)

    rotation_deg = np.deg2rad(rotation_deg).tolist()

    vector = euler_rotate(vector, rotation_deg)

    vector = euler_rotate(vector, ROTATION_OFFSET)

    intersect: npt.NDArray[np.float64] = plane_collision(vector, height)

    return intersect


def plane_collision(
    ray_direction: npt.NDArray[np.float64], height: float
) -> npt.NDArray[np.float64]:
    """
    Returns the point where a ray intersects the XY plane

    Parameters
    ----------
    ray_direction : npt.NDArray[np.float64]
        XYZ coordinates that represent the direction a ray faces from (0, 0, 0)
    height : float
        The Z coordinate for the starting height of the ray; can be any units

    Returns
    -------
    intersect : npt.NDArray[np.float64]
        The ray's intersection with the plane in [X,Y] format

    """
    # Find the "time" at which the line intersects the plane
    # Line is defined as ray_direction * time + origin.
    # Origin is the point at X, Y, Z = (0, 0, height)

    time: np.float64 = -height / ray_direction[2]
    intersect: npt.NDArray[np.float64] = ray_direction[:2] * time

    return intersect


def pixel_vector(
    pixel: Tuple[int, int], image_shape: Tuple[int, ...], focal_length: float
) -> npt.NDArray[np.float64]:
    """
    Generates a vector representing the given pixel.
    Pixels are in row-major form [Y, X] to match numpy indexing.

    Parameters
    ----------
    pixel : Tuple[int, int]
        The coordinates of the pixel in [Y, X] form
    image_shape : Tuple[int, int, int]
        The shape of the image (returned by image.shape when image is a numpy image array)
    focal_length : float
        The camera's focal length - used to generate the camera's fields of view

    Returns
    -------
    pixel_vector : npt.NDArray[np.float64]
        The vector that represents the direction of the given pixel
    """

    # Find the FOVs using the focal length
    fov_h: float
    fov_v: float
    fov_h, fov_v = focal_length_to_fovs(focal_length)

    return camera_vector(
        pixel_angle(fov_h, pixel[1] / image_shape[1]),
        pixel_angle(fov_v, pixel[0] / image_shape[0]),
    )


def pixel_angle(fov: float, ratio: float) -> float:
    """
    Calculates a pixel's angle from the center of the camera on a single axis. Analogous to the
    pixel's "fov"

    Only one component of the pixel is used here, call this function for each X and Y

    Parameters
    ----------
    fov : float
        The field of view of the camera in radians olong a given axis
    ratio : float
        The pixel's position as a ratio of the coordinate to the length of the image
        Example: For an image that is 1080 pixels wide, a pixel at position 270 would have a
        ratio of 0.25

    Returns
    -------
    angle : float
        The pixel's angle from the center of the camera along a single axis
    """
    return np.arctan(np.tan(fov / 2) * (1 - 2 * ratio))


def focal_length_to_fovs(focal_length: float) -> Tuple[float, float]:
    """
    Converts a given focal length to the horizontal and vertical fields of view in radians

    Uses SENSOR_WIDTH and SENSOR_HEIGHT, which are set to 13.2 and 8.8 respectively, the size of
    the sensor in the Sony RX100 vii

    Parameters
    ----------
    focal_length: float
        The focal length of the camera in millimeters
    Returns
    -------
    fields_of_view : Tuple[float, float]
        The fields of view in radians
        Format is [horizontal, vertical]
    """
    return get_fov(focal_length, SENSOR_WIDTH), get_fov(focal_length, SENSOR_HEIGHT)


def get_fov(focal_length: float, sensor_size: float) -> float:
    """
    Converts a given focal length and sensor length to the corresponding field of view in radians

    Parameters
    ----------
    focal_length : float
        The focal length of the camera in millimeters
    sensor_size:
        The sensor size along one axis in millimeters

    Returns
    -------
    fov : float
        The field of view in radians
    """

    return 2 * np.arctan(sensor_size / (2 * focal_length))


def camera_vector(h_angle: float, v_angle: float) -> npt.NDArray[np.float64]:
    """
    Generates a vector with an angle h_angle with the horizontal and an angle v_angle with the
    vertical.

    Using camera fovs will generate a vector that represents the corner of the camera's view.

    Parameters
    ----------
    h_angle : float
        The angle in radians to rotate horizontally
    v_angle : float
        The angle in radians to rotate vertically
    Returns
    -------
    camera_vector : npt.NDArray[np.float64]
        The vector which represents a given location in an image
    """

    # Calculate the vertical rotation needed for the final vector to have the desired direction
    edge: float = edge_angle(v_angle, h_angle)

    vector: npt.NDArray[np.float64] = np.array([1, 0, 0], dtype=np.float64)
    return euler_rotate(vector, [0, edge, -h_angle])


def edge_angle(v_angle: float, h_angle: float) -> float:
    """
    Finds the angle such that rotating by edge_angle on the Y axis then rotating by h_angle on
    the Z axis gives a vector an angle v_angle with the Y axis

    Can be derived using a square pyramid of height 1

    Parameters
    ----------
    v_angle : float
        The vertical angle
    h_angle : float
        The horizontal angle
    Returns
    -------
    edge_angle : float
        The angle to rotate vertically
    """

    return np.arctan(np.tan(v_angle) * np.cos(h_angle))


def euler_rotate(
    vector: npt.NDArray[np.float64], rotation_deg: List[float]
) -> npt.NDArray[np.float64]:
    """
    Rotates a vector based on a given roll, pitch, and yaw.

    Follows the MAVSDK.EulerAngle convention - positive roll is banking to the right, positive
    pitch is pitching nose up, positive yaw is clock-wise seen from above.

    Parameters
    ----------
    vector: npt.NDArray[np.float64]
        A vector represented by an XYZ coordinate that will be rotated
    rotation_deg: List[float]
        The [roll, pitch, yaw] rotation in radians
    Returns
    -------
    rotated_vector : npt.NDArray[np.float64]
        The vector which has been rotated
    """

    # Reverse the Y and Z rotation to match MAVSDK convention
    rotation_deg[1] *= -1
    rotation_deg[2] *= -1

    return Rotation.from_euler("xyz", rotation_deg).apply(vector)