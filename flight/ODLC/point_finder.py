import utm
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from typing import List, Dict, Tuple


def latlon_to_utm(coords: Dict) -> Dict:
    """
    Converts latlon coordinates to utm coordinates and adds the data to the dictionary

    Args:
        coords (dict): A dictionary containing lat long coordinates

    Returns:
        dict: An updated dictionary with additional keys and values with utm data
    """
    utm_coords = utm.from_latlon(coords["latitude"], coords["longitude"])
    coords["utm_x"] = utm_coords[0]
    coords["utm_y"] = utm_coords[1]
    coords["utm_zone_number"] = utm_coords[2]
    coords["utm_zone_letter"] = utm_coords[3]
    return coords


def all_latlon_to_utm(list_of_coords: List[Dict]) -> List[Dict]:
    """
    Converts a list of dictionarys with latlon data to add utm data

    Args:
        list_of_coords (list[dict]): A list of dictionaries that contain lat long data

    Returns:
        list[dict]: An updated list of dictionaries with added utm data
    """
    for i in range(len(list_of_coords)):
        list_of_coords[i] = latlon_to_utm(list_of_coords[i])
    return list_of_coords


def scale_polygon(my_polygon, scale_factor=0.1, enlarge=False) -> Polygon:
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * scale_factor

    if enlarge:
        my_polygon_resized = my_polygon.buffer(shrink_distance)  # enlarge
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

    return my_polygon_resized


def find_closest_point(odlc: Dict, shrunk_boundary: List[Dict], obstacles: List[Dict]) -> Tuple:
    poly_points = []
    for point in shrunk_boundary:
        poly_points.append((point["utm_x"], point["utm_y"]))

    boundary_shape = Polygon(poly_points)
    odlc_shape = Point(odlc["utm_x"], odlc["utm_y"])

    for obstacle in obstacles:
        # create obstacle as shapely shape
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]
        circle = Point(x, y).buffer(radius).boundary
        obstacle_shape = Polygon(circle)

        # remove obstacle area from boundary polygon
        boundary_shape = boundary_shape.difference(obstacle_shape)

    # scale down boundary to add a saftey margin
    boundary_shape = scale_polygon(boundary_shape)

    p1, p2 = nearest_points(
        boundary_shape, odlc_shape
    )  # point returned in same order as input shapes

    closest_point = p1

    zone_number = odlc["utm_zone_number"]
    zone_letter = odlc["utm_zone_letter"]

    return (
        {
            "utm_x": closest_point.x,
            "utm_y": closest_point.y,
            "utm_zone_number": zone_number,
            "utm_zone_letter": zone_letter,
            "latitude": utm.to_latlon(closest_point.x, closest_point.y, zone_number, zone_letter)[
                0
            ],
            "longitude": utm.to_latlon(closest_point.x, closest_point.y, zone_number, zone_letter)[
                1
            ],
        },
        list(zip(*boundary_shape.exterior.coords.xy)),
    )
