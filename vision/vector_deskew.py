import numpy as np
import numpy.typing as npt
import cv2
import mavsdk

import coordinate_lengths
from vector_utils import *


def poly_area(coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    i = np.arange(len(x))

    # An implementation of the shoelace algorithm using numpy functions
    area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)

    return area


def pixel_distance(pixel1, pixel2, image_shape, focal_length, attitude, position):
    height = position.relative_altitude_m

    intersect1 = pixel_intersect(pixel1, image_shape, focal_length, attitude, height)
    intersect2 = pixel_intersect(pixel2, image_shape, focal_length, attitude, height)

    # Calculate the distance between the two intersects
    distance = np.linalg.norm(intersect1 - intersect2)

    distance = 3.28084 * distance  # convert meters to feet?

    return distance


def pixel_coords(pixel, image_shape, focal_length, attitude, position):
    height = position.relative_altitude_m

    # Calculate the latitude and longitude lengths (in meters)
    latitude_length = coordinate_lengths.latitude_length(position.latitude_deg)
    longitude_length = coordinate_lengths.longitude_length(position.latitude_deg)

    # Find the pixel's intersect with the ground to get the location relative to the drone
    intersect = pixel_intersect(pixel, image_shape, focal_length, attitude, height)

    # Invert the X axis so that the longitude is correct
    intersect[1] *= -1

    # Convert the location to latitude and longitude and add it to the drone's coordinates
    pixel_lat = position.latitude_deg + intersect[0] / latitude_length
    pixel_lon = position.longitude_deg + intersect[1] / longitude_length

    return pixel_lat, pixel_lon


def vector_deskew(image, focal_length, attitude: mavsdk.telemetry.EulerAngle, area_scale=1,
                  interpolation=cv2.INTER_NEAREST):
    orig_height, orig_width, _ = image.shape

    src_pts = np.float32(
        [
            [0, 0],
            [orig_width, 0],
            [orig_width, orig_height],
            [0, orig_height]
        ]
    )

    # Convert XY to YX
    flipped = np.flip(src_pts, axis=1)

    intersects = np.float32([pixel_intersect(point, image.shape, focal_length, attitude)
                             for point in flipped])

    # Flip the endpoints over the X axis (top left is 0,0 for images)
    intersects[:, 1] *= -1

    # Subtract the minimum on both axes so the minimum values on each axis are 0
    intersects = intersects - intersects.min(axis=0)

    # Find the area of the resulting shape
    area = poly_area(intersects)

    # Scale the output so the area of the important pixels is about the same as the starting image
    target_area = image.shape[0] * image.shape[1] * area_scale
    scale = np.sqrt(target_area / area)
    dst_pts = intersects * scale

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    result_height = int(dst_pts[:, 1].max()) + 1
    result_width = int(dst_pts[:, 0].max()) + 1

    result = cv2.warpPerspective(
        image, matrix, (result_width, result_height), flags=interpolation,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    return result


def main():
    # coords = pixel_coords(
    #     [720/2, 1080/2], [720, 1080, 3], 10,
    #     mavsdk.telemetry.EulerAngle(0, -45, 180, 0),
    #     mavsdk.telemetry.Position(0, 0, None, 100000)
    # )
    #
    # print(np.around(coords, decimals=8))

    image = cv2.imread("render2.png")
    # print(type(image.shape))
    # image = np.dstack((image, np.full(image.shape[:2], 255)))
    # image = feather_edges(image, 100)
    image = vector_deskew(image, 10, mavsdk.telemetry.EulerAngle(0, -30, -30, 0))

    # image = image[:, :, :3]


if __name__ == "__main__":
    main()
