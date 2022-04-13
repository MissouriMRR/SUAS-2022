"""
Functions for generating a search path for standard odlc objects
"""

from typing import List, Dict, Tuple

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import utm


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
    """Converts a list of dictionaries with latlon data to add utm data

    Parameters
    ----------
    list_of_coords : List[Dict[str, float]]
        A list of dictionaries that contain lat long data

    Returns
    -------
    List[Dict[str, float]]
        list[dict]: An updated list of dictionaries with added utm data
    """

    for i, _ in enumerate(list_of_coords):
        list_of_coords[i] = latlon_to_utm(list_of_coords[i])
    return list_of_coords


# use height/2 of camera image as buffer distance
def generate_search_path(search_area_points, buffer_distance):
    poly_points = [(point["utm_x"], point["utm_y"]) for point in search_area_points]

    boundary_shape = Polygon(poly_points)

    search_paths = []
    search_paths.append(
        list(zip(*boundary_shape.exterior.coords.xy))  # pylint: disable=maybe-no-member
    )

    while True:
        boundary_shape = boundary_shape.buffer(buffer_distance, single_sided=True)
        if boundary_shape.area <= 0:
            break
        search_paths.append(
            list(zip(*boundary_shape.exterior.coords.xy))  # pylint: disable=maybe-no-member
        )

    return search_paths
