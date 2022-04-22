#!/user/bin/python

import math, time
import numpy as np
from typing import Tuple, List

# ------ GLOBAL VARIABLES ------ #
# Units: mm
SENSOR_W: float = 4.8
SENSOR_H: float = 3.6
SENSOR_D: float = 6.0

# Units: percent (0.0 -> 1.0)
OVERLAP: float = 0.1

# Radius of Earth
# Units: ft
RADIUS: float = 20902230.971129

# Bearings for different directions
SOUTH: float = math.pi
EAST: float = 1.570796
NW: float = 5.585054

# ------ GLOBAL VARIABLES ------ #


def map(
    info: Tuple[Tuple[float, float], float],
    altitude: int = 750,
    focal_length: float = 4.9,
) -> List[Tuple[float, float]]:
    """
    Calculates the map area, cuts map area into sections,
    then creates flight path for drone through sections.

    Parameters
    ----------
    info: Tuple[Tuple[float, float], float]
        The center location in latitude, longitude of the map, and then height of the map
    altitude: int
    focal_length: float

    Returns
    -------
    path: List[Tuple[float, float]]
        The path that the drone will take through the map
    """

    # Takes in tuple(start location, map height), altitude, and focal_length]
    # Assumes altitude and focal_length doesn't change
    # Returns flight path of drone to snap images of map

    # Split info into center of map location and map height
    center_map: Tuple[float, float] = info[0]
    map_h: float = info[1]

    # Calculate the width and height of camera image
    cam_w, cam_h = image_area(altitude, focal_length)
    #print("Camera Width:", cam_w)
    #print("Camera Height:", cam_h)

    # Determine map width using MAP_H and aspect ratio 16:9
    # Units: ft
    map_w: float = (16 * map_h) / 9

    #print("Map Width:", map_w)
    #print("Map Height:", map_h)

    # Find center of each piece consider OVERLAP
    ft_waypoints: List[Tuple[float, float]] = find_centers(
        map_w, map_h, cam_w, cam_h, OVERLAP
    )[0]
    col = find_centers(map_w, map_h, cam_w, cam_h, OVERLAP)[1]
    #print("Num of waypoints:", len(ft_waypoints))

    # Convert centers to lat, long coordinates
    coord: List[Tuple[float, float]] = coordinates(ft_waypoints, info[0], map_w, map_h)
    #print("Coord:", len(coord))

    # Flight path algorithm
    path = flight_path(coord, col)

    return path


def image_area(altitude: int, focal_length: float) -> Tuple[float, float]:
    """
    Takes in the altitude of the drone and the zoom (focal_length) in order to determine
    the area of the image captures by the drone

    Parameters
    ----------
    altitude: int
    focal_length: float

    Returns
    -------
    (cam_w, cam_h): Tuple[float, float]

    Notes
    -----
    fov is the Field of View of the image area
    fovH is the horizontal field of view distance
    fovV is the vertical field of view distance
    """

    fov: float = 2 * math.atan((SENSOR_W / (2 * focal_length)))
    #print("FOV:", fov)

    fovH: float = SENSOR_W / SENSOR_D * fov
    fovV: float = SENSOR_H / SENSOR_D * fov

    #print("fovH:", fovH)
    #print("fovV:", fovV)

    cam_w: float = 2 * altitude * math.tan(fovH / 2)
    cam_h: float = 2 * altitude * math.tan(fovV / 2)

    return (cam_w, cam_h)


def find_centers(
    map_w: float,
    map_h: float,
    cam_w: float,
    cam_h: float,
    overlap: float,
) -> List[Tuple[float, float]]:
    """
    This function will split the map into even sections accounting for the image area
    and overlap for image stitching. It will start at the top left corner and move
    a determined distance from from the top left corner to the center of a section
    for every section.

    Parameters
    ----------
    map_w: float
    map_h: float
    cam_w: float
    cam_h: float,
    overlap: float

    Returns
    -------
    waypoints: List[Tuple[float, float]]
    """

    # Return a 2d list of center locations for each image
    # List of way points
    waypoints: List[Tuple[float, float]] = []

    # Determines how much per section we need to move from current section over
    widthMove: float = cam_w - (cam_w * overlap)
    heightMove: float = cam_h - (cam_h * overlap)

    # Adds starting waypoint to list of waypoints as (x,y)
    midArea = 1 - (2 * overlap)
    x: float = (cam_w * midArea) / 2
    y: float = (cam_h * midArea) / 2

    # Find number of rows and cols for map splitting
    rows: int = math.ceil((map_w - x) / widthMove) + 1
    cols: int = math.ceil((map_h - y) / heightMove) + 1

    #print("Rows:", rows)
    #print("Cols:", cols)

    # Find all waypoints for each section
    for i in range(rows):
        for j in range(cols):
            waypoints.append((x + (i * widthMove), y + (j * heightMove)))

    return waypoints, cols


def coordinates(
    waypoints: List[Tuple[float, float]],
    center: Tuple[float, float],
    map_w: float,
    map_h: float,
) -> List[Tuple[float, float]]:

    """
    Finds the coordinate locations of each section of the waypoints converting
    from distance traveled and bearing to lat,long coordinates.

    Parameters
    ----------
    waypoints: List[Tuple[float, float]]
    center: Tuple[float, float]
    map_w: float
    map_h: float

    Returns
    -------
    coord: List[Tuple[float, float]]
    """

    # Converts feet coordinates to lat, long coordinates
    # Returns list of waypoints in lat, long coordinates
    coord: List[Tuple[float, float]] = []

    # First calculate where the (0,0) coordinate is in lat,long
    # Move (0,0) is the top right corner so we move left and up to get there
    base: float = map_w / 2
    height: float = map_h / 2
    dist: float = math.hypot(height, base)
    start: Tuple[float, float] = conversion(center, NW, dist)

    # Convert rest of waypoints using start: lat, long as origin
    for point in waypoints:
        move_x: Tuple[float, float] = conversion(start, EAST, point[0])
        loc: Tuple[float, float] = conversion(start, SOUTH, point[1])
        coord.append(loc)

    return coord


def conversion(
    coord: Tuple[float, float], bearing: float, distance: float
) -> Tuple[float, float]:
    """
    Uses the current location, bearing of travel, and the distance traveled in order to determine
    the lat,long coordinates of the new location

    Parameters
    ----------
    coord: Tuple[float, float]
    bearing: float
    distance: float

    Returns
    -------
    (new_lat: float, new_long: float)

    References
    ----------
    https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing

    This is the link to the reference used for the conversion of the parameters into lat,long coordinates
    """
    o_lat: float = math.radians(coord[0])  # Original lat point converted to radians
    o_lon: float = math.radians(coord[1])  # Original long point converted to radians

    new_lat: float = math.asin(
        math.sin(o_lat) * math.cos(distance / RADIUS)
        + math.cos(o_lat) * math.sin(distance / RADIUS) * math.cos(bearing)
    )

    new_lon: float = o_lon + math.atan2(
        math.sin(bearing) * math.sin(distance / RADIUS) * math.cos(o_lat),
        math.cos(distance / RADIUS) - math.sin(o_lat) * math.sin(new_lat),
    )

    new_lat: float = math.degrees(new_lat)
    new_lon: float = math.degrees(new_lon)

    return (new_lat, new_lon)


def flight_path(points: List[Tuple[float, float]], col: int) -> List[Tuple[float, float]]:
    new_arr = []
    temp = []
    for loc in points:
        temp.append(loc)
        if len(temp) == col:
            new_arr.append(temp)
            temp = []
    
    for i in range(len(new_arr)):
        if i % 2 != 0:
            new_arr[i].reverse()
    
    path = []
    for row in new_arr:
        path += row

    #print(path)

    return path

# if __name__ == "__main__":
#     start_time = time.time()
#     info = ((38.145103, -76.427856), 1200.0)
#     map(info)
#     print("%.4f seconds" % (time.time() - start_time))
