import numpy.typing as npt
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
import mavsdk

# Sony RX100 vii sensor size
SENSOR_WIDTH = 13.2
SENSOR_HEIGHT = 8.8


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


def focal_length_to_fovs(
        focal_length: float,
        sensor_size: Optional[Tuple[float, float]] = (SENSOR_WIDTH, SENSOR_HEIGHT)
) -> Tuple[float, float]:
    """
    Converts a given focal length to the horizontal and vertical fields of view in radians

    Parameters
    ----------
    focal_length: float
        The focal length of the camera in millimeters
    sensor_size: Optional[Tuple[float, float]]
        The dimensions (width, height) of the sensor. Defaults to SENSOR_WIDTH and SENSOR_HEIGHT,
        which are set to 13.2 and 8.8 respectively, the size of the sensor in the Sony RX100 vii
    Returns
    -------
    fields_of_view : Tuple[float, float]
        The horizontal and vertical fields of view in radians
    """
    return get_fov(focal_length, sensor_size[0]), get_fov(focal_length, sensor_size[1])


# This gets the actual angle of the edge of the camera view; this can be derived using a square
# pyramid with height 1
def edge_angle(horizontal_angle: float, vertical_angle: float) -> float:
    """
    Finds the angle needed to rotate

    Parameters
    ----------
    horizontal_angle
    vertical_angle

    Returns
    -------

    """
    return np.arctan(np.tan(horizontal_angle) * np.cos(vertical_angle))


# Calculates the other angle if one FOV is known and the other isn't (DELETE THIS)
def find_angle(angle: float, aspect_ratio: float) -> float:
    return 2 * np.arctan(aspect_ratio * np.tan(angle / 2))


def plane_collision(
        ray_direction: npt.NDArray[np.float64],
        height: float = 1,
        epsilon: float = 1e-6
) -> npt.NDArray[np.float64]:
    """
    Returns the point where a ray intersects the XY plane

    Parameters
    ----------
    ray_direction : npt.NDArray[np.float64]
        XYZ coordinates that represent the direction a ray faces from (0, 0, 0)
    height : float
        The Z coordinate for the starting height of the ray; can be any units
    epsilon : float
        Minimum value for the dot product of the ray direction and plane normal

    Raises
    ------
    RuntimeError: "no intersection or line is parallel to plane"
        Occurs when the ray direction is facing away from or parallel to the plane

    References
    ----------
    http://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    """

    # Define the direction of the side face of the plane (In this case, facing upwards towards +Z)
    plane_normal: npt.NDArray[np.float64] = np.array([0, 0, 1])

    plane_point: npt.NDArray[np.float64] = np.array([0, 0, 0])  # Any point on the plane
    ray_point: npt.NDArray[np.float64] = np.array([0, 0, height])  # Origin point of the ray

    ndotu: np.float64 = plane_normal.dot(ray_direction)

    # Checks to make sure the ray is pointing into the plane
    if -ndotu < epsilon:
        raise RuntimeError("no intersection or line is parallel to plane")

    # I didn't make this math but it works
    w: npt.NDArray[np.int64] = ray_point - plane_point
    si: np.float64 = -plane_normal.dot(w) / ndotu
    psi: npt.NDArray[np.float64] = w + si * ray_direction + plane_point

    psi = np.delete(psi, -1)  # Remove the Z coordinate since it's always 0
    return psi


def euler_rotate(
        vector: npt.NDArray[np.float64],
        rotation: List[float]
) -> npt.NDArray[np.float64]:
    """
    Rotates a vector based on a given roll, pitch, and yaw.

    Follows the MAVSDK.EulerAngle convention - positive roll is banking to the right, positive
    pitch is pitching nose up, positive yaw is clock-wise seen from above.

    Parameters
    ----------
    vector: npt.NDArray[np.float64]
        A vector represented by an XYZ coordinate that will be rotated
    rotation: List[float]
        The [roll, pitch, yaw] rotation in radians
    Returns
    -------
    rotated_vector : npt.NDArray[np.float64]
        The vector which has been rotated
    """

    # Reverse the Y and Z rotation to match MAVSDK convention
    rotation[1] *= -1
    rotation[2] *= -1

    return R.from_euler('xyz', rotation).apply(vector)


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


def pixel_vector(
        pixel: Tuple[int, int],
        image_shape: Tuple[int, int, int],
        focal_length: float
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
        pixel_angle(fov_v, pixel[0] / image_shape[0])
    )


def pixel_intersect(
        pixel: Tuple[int, int],
        image_shape: Tuple[int, int, int],
        focal_length: float,
        attitude: mavsdk.telemetry.EulerAngle,
        height: Optional[float] = 1
) -> npt.NDArray[np.float64]:
    """
    Finds the intersection [X,Y] of a given pixel with the ground.
    A camera with no rotation points in the +X direction and is centered at [0, 0, height].

    Parameters
    ----------
    pixel : Tuple[int, int]
        The coordinates of the pixel in [Y, X] form
    image_shape : Tuple[int, int, int]
        The shape of the image (returned by image.shape when image is a numpy image array)
    focal_length : float
        The camera's focal length
    attitude : mavsdk.telemetry.EulerAngle
        The rotation of the drone given by MAVSDK
        For testing purposes, you can generate an EulerAngle object as following:
        mavsdk.telemetry.EulerAngle(roll_deg, pitch_deg, yaw_deg, 0)
        With 0 as the input for the timestamp which is not needed.
    height : Optional[float]
        The height of the drone in any units. If a height is given, the units of the output will
        be the units of the input. Defaults to 1.
    Returns
    -------
    intersect : npt.NDArray[np.float64]
        The coordinates [X,Y] where the pixel's vector intersects with the ground.
    """

    # Create the normalized vector representing the direction of the given pixel
    vector: npt.NDArray[np.float64] = pixel_vector(pixel, image_shape, focal_length)

    # Extract the values from the EulerAngle object
    cam_roll: float = np.deg2rad(attitude.roll_deg)
    cam_pitch: float = np.deg2rad(attitude.pitch_deg)
    cam_yaw: float = np.deg2rad(attitude.yaw_deg)

    vector = euler_rotate(vector, [cam_roll, cam_pitch, cam_yaw])

    intersect: npt.NDArray[np.float64] = plane_collision(vector, height)

    return intersect

# TODO:
#   Specify radians for each
