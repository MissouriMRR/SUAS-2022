"""Helper functions for converting and handling data"""

from typing import List, Tuple, Dict
import utm
from shapely.geometry import Point, Polygon


def latlon_to_utm(coords: Dict[str, float]) -> Dict[str, float]:
    """Converts latlon coordinates to utm coordinates and adds the data to the dictionary

    Parameters
    ----------
    coords : Dict[str, float]
        A dictionary containing lat long coordinates

    Returns
    -------
    Dict[str, float]
        An updated dictionary with additional keys and values with utm data
    """

    utm_coords = utm.from_latlon(coords["latitude"], coords["longitude"])
    coords["utm_x"] = utm_coords[0]
    coords["utm_y"] = utm_coords[1]
    coords["utm_zone_number"] = utm_coords[2]
    coords["utm_zone_letter"] = utm_coords[3]
    return coords


def all_latlon_to_utm(list_of_coords: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Converts a list of dictionarys with latlon data to add utm data

    Parameters
    ----------
    list_of_coords : List[Dict[str, float]]
        A list of dictionaries that contain lat long data

    Returns
    -------
    List[Dict[str, float]]
        An updated list of dictionaries with added utm data
    """

    for i, _ in enumerate(list_of_coords):
        list_of_coords[i] = latlon_to_utm(list_of_coords[i])
    return list_of_coords


def coords_to_shape(coords: List[Dict[str, float]]) -> Polygon:
    """Converts a list of dictionary location data to a shapely polygon

    Parameters
    ----------
    coords : List[Dict[str, float]]
        A list of dictionaries that contain utm data

    Returns
    -------
    Polygon
        A shapely polygon from the given utm data
    """

    poly_coords = [(point["utm_x"], point["utm_y"]) for point in coords]
    shape = Polygon(poly_coords)
    return shape


def circles_to_shape(circles: List[Dict[str, float]]) -> List[Point]:
    """Converts a list of circles to shapely shapes

    Parameters
    ----------
    circles : List[Dict[str, float]]
        A list of dictionaries that contain central utm coordinates and a radius

    Returns
    -------
    List[Point]
        A point, enlarged to the given radius
    """

    circle_shapes: List[Point] = []
    for circle in circles:
        x = circle["utm_x"]
        y = circle["utm_y"]
        radius = circle["radius"]
        circle_shape = Point(x, y).buffer(radius).boundary
        circle_shapes.append(circle_shape)
    return circle_shapes


def coords_to_points(coords: List[Dict[str, float]]) -> List[Tuple[Point, float]]:
    """Converts a list of utm coordinates to a list of shapely points

    Parameters
    ----------
    coords : List[Dict[str, float]]
        A list of coordinates with utm position and altitude data

    Returns
    -------
    List[Tuple[Point, float]]
        A list of points with altitudes
    """

    points: List[Tuple[Point, float]] = []
    for coord in coords:
        point = Point(coord["utm_x"], coord["utm_y"])
        alt = coord["altitude"]
        points.append((point, alt))
    return points


def all_feet_to_meters(
    obstacles: List[Dict[str, float]], obstacle_buffer: int
) -> List[Dict[str, float]]:
    """Converts obstacle radius and height to meters, then adds
    a buffer to radius and height

    Parameters
    ----------
    obstacles : List[Dict[str, float]]
        A list of obstacles in dictionary format
    obstacle_buffer : int
        A buffer amount in meters

    Returns
    -------
    List[Dict[str, float]]
        An updated list in the original format with units of meters instead of feet
    """

    feet_to_meters_multiplier: float = 0.3048
    for obstacle in obstacles:
        # conversion
        obstacle["radius"] *= feet_to_meters_multiplier
        obstacle["height"] *= feet_to_meters_multiplier
        # buffer
        obstacle["radius"] += obstacle_buffer
        obstacle["height"] += obstacle_buffer
    return obstacles


def path_to_latlon(
    path: List[Tuple[Point, float]], zone_num: int, zone_letter: str
) -> List[Tuple[float, float, float]]:
    """_summary_

    Parameters
    ----------
    path : List[Tuple[Point, float]]
        _description_
    zone_num : int
        _description_
    zone_letter : str
        _description_

    Returns
    -------
    List[Tuple[float, float, float]]
        _description_
    """

    gps_path: List[Tuple[float, float, float]] = []
    for loc in path:
        point: Tuple[float, float] = utm.to_latlon(loc[0].x, loc[0].y, zone_num, zone_letter)
        alt: float = loc[1]
        gps_path.append((*point, alt))
    return gps_path


def get_zone_info(boundary: List[Dict[str, float]]) -> Tuple[int, str]:
    """Gets the utm zone number and letter based off the first point in the boundary

    Parameters
    ----------
    boundary : List[Dict[str, float]]
        Boundary data in dictionary format

    Returns
    -------
    Tuple[int, str]
        The utm zone number and letter
    """

    return (int(boundary[0]["utm_zone_number"]), str(boundary[0]["utm_zone_letter"]))
