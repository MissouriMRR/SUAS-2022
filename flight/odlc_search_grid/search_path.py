"""
Functions for generating search paths to cover an area for finding the standard odlc objects
"""

from typing import List, Dict, Tuple
from shapely.geometry import Polygon
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


def generate_search_paths(
    search_area_points: List[Dict[str, float]], buffer_distance: int
) -> List[Tuple[float, float]]:
    """Generates a list of search paths of increasingly smaller sizes until the whole area
    of the original shape has been covered

    Parameters
    ----------
    search_area_points : Dict[str, float]
        A list of coordinates in dictionary form that contain utm coordinate data
    buffer_distance : int
        The distance that each search path will be apart from the previous one.
        For complete photographic coverage of the area, this should be equal to half the height
        of the area the camera covers on the ground given the current altitude.

    Returns
    -------
    List[Tuple[float, float]]
        A list of concentric search paths that cover the area of the polygon
    """

    # convert to shapely polygon for buffer operations
    poly_points = [(point["utm_x"], point["utm_y"]) for point in search_area_points]
    boundary_shape = Polygon(poly_points)

    search_paths = []

    # shrink boundary by a fixed amount until the area it covers is 0
    # add the smaller boundary to our list of search paths on each iteration
    while boundary_shape.area > 0:
        search_paths.append(
            tuple(zip(*boundary_shape.exterior.coords.xy))  # pylint: disable=maybe-no-member
        )
        boundary_shape = boundary_shape.buffer(buffer_distance, single_sided=True)

    return search_paths
