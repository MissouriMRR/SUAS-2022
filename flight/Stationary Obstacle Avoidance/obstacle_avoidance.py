from typing import List, Tuple
from collections import namedtuple
import utm
from fractions import Fraction
from shapely.geometry import LineString
from shapely.geometry import Point
import math_functions
import plotter


def latlon_to_utm(coords):
    utm_coords = utm.from_latlon(coords["latitude"], coords["longitude"])
    coords["utm_x"] = utm_coords[0]
    coords["utm_y"] = utm_coords[1]
    coords["utm_zone_number"] = utm_coords[2]
    coords["utm_zone_letter"] = utm_coords[3]
    return coords


def all_latlon_to_utm(list_of_coords):
    for i in range(len(list_of_coords)):
        list_of_coords[i] = latlon_to_utm(list_of_coords[i])
    return list_of_coords


def check_for_collision(p1, p2, obstacles, padding):
    for i, obstacle in enumerate(obstacles):
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]
        line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        circle = (
            Point(x, y).buffer(radius + padding).boundary
        )  # https://stackoverflow.com/a/30998492
        if circle.intersection(line):
            return i
    else:
        return None


def find_new_point(
    pointA, pointB, obstacle, radius, padding, max_distance=0, debugging=False
) -> tuple:
    # rough calculation of new height
    new_altitude = (pointA[2] + pointB[2]) / 2

    # find tangents to the circle from points A and B
    (tangentA1, tangentA2) = math_functions.tangent_points(
        pointA, obstacle, radius + padding
    )
    (tangentB1, tangentB2) = math_functions.tangent_points(
        pointB, obstacle, radius + padding
    )

    # get lines between points A, B, and their respective tangents to the circle
    (
        route1_segment1,
        route1_segment2,
        route2_segment1,
        route2_segment2,
    ) = math_functions.resolve_closest_tangents(
        pointA, pointB, tangentA1, tangentA2, tangentB1, tangentB2
    )

    # find intersection points between lines
    route1X, route1Y = math_functions.intersection(route1_segment1, route1_segment2)
    route2X, route2Y = math_functions.intersection(route2_segment1, route2_segment2)

    # calculate distance of each route on either side of circle
    route1_distance = math_functions.distance(
        pointA, (route1X, route1Y)
    ) + math_functions.distance(pointB, (route1X, route1Y))
    route2_distance = math_functions.distance(
        pointA, (route2X, route2Y)
    ) + math_functions.distance(pointB, (route2X, route2Y))

    # choose shortest path
    if route1_distance < route2_distance:
        new_point = (route1X, route1Y)
        alt_point = (route2X, route2Y)
    else:
        new_point = (route2X, route2Y)
        alt_point = (route1X, route1Y)

    # Handle scenario where new waypoint is too far from the circle
    special_case = False
    # temp_x, extra1X, extra2X, temp_y, extra1Y, extra2Y = 0,0,0,0,0,0 # init vars
    waypoint_distance_from_circle = math_functions.distance(
        (new_point[0], new_point[1]), obstacle
    )
    if waypoint_distance_from_circle > radius + padding + max_distance:
        special_case = True

        print("New waypoint is too far away, splitting and adding more waypoints")
        # https://math.stackexchange.com/a/1630886
        t = (radius + padding) / waypoint_distance_from_circle
        temp_x = ((1 - t) * obstacle[0]) + (t * new_point[0])
        temp_y = ((1 - t) * obstacle[1]) + (t * new_point[1])
        temp_point = (temp_x, temp_y)

        first_segment = math_functions.line(
            (pointA[0], pointA[1]), (new_point[0], new_point[1])
        )
        second_segment = math_functions.line(
            (pointB[0], pointB[1]), (new_point[0], new_point[1])
        )

        m = -1 / (
            (temp_y - obstacle[1]) / (temp_x - obstacle[0])
        )  # slope of new middle line segment
        m_frac = Fraction(str(m))
        slopeX = m_frac.denominator
        slopeY = m_frac.numerator
        middle_segment = math_functions.line(
            (temp_x, temp_y), (temp_x + slopeX, temp_y + slopeY)
        )

        extra1X, extra1Y = math_functions.intersection(first_segment, middle_segment)
        extra2X, extra2Y = math_functions.intersection(second_segment, middle_segment)
        extra_1 = (extra1X, extra1Y)
        extra_2 = (extra2X, extra2Y)

    # plot extra data about this specific path and obstacle
    if debugging:
        plotter.plot_debug(
            pointA,
            pointB,
            obstacle,
            tangentA1,
            tangentA2,
            tangentB1,
            tangentB2,
            new_point,
            alt_point,
            temp_point,
            extra_1,
            extra_2,
            special_case,
            padding,
        )

    zone_number = pointA[3]
    zone_letter = pointA[4]

    if special_case:
        return [
            {
                "utm_x": extra_1[0],
                "utm_y": extra_1[1],
                "utm_zone_number": zone_number,
                "utm_zone_letter": zone_letter,
                "latitude": utm.to_latlon(*extra_1, zone_number, zone_letter)[0],
                "longitude": utm.to_latlon(*extra_1, zone_number, zone_letter)[1],
                "altitude": new_altitude,
            },
            {
                "utm_x": extra_2[0],
                "utm_y": extra_2[1],
                "utm_zone_number": zone_number,
                "utm_zone_letter": zone_letter,
                "latitude": utm.to_latlon(*extra_2, zone_number, zone_letter)[0],
                "longitude": utm.to_latlon(*extra_2, zone_number, zone_letter)[1],
                "altitude": new_altitude,
            },
        ]
    else:
        return [
            {
                "utm_x": new_point[0],
                "utm_y": new_point[1],
                "utm_zone_number": zone_number,
                "utm_zone_letter": zone_letter,
                "latitude": utm.to_latlon(*new_point, zone_number, zone_letter)[0],
                "longitude": utm.to_latlon(*new_point, zone_number, zone_letter)[1],
                "altitude": new_altitude,
            }
        ]


def get_safe_route(waypoints, obstacles, padding, max_distance, debugging=False):
    i = 0
    num_waypoints = len(waypoints) - 1
    while i < num_waypoints:
        point_a = (
            waypoints[i]["utm_x"],
            waypoints[i]["utm_y"],
            waypoints[i]["altitude"],
            waypoints[i]["utm_zone_number"],
            waypoints[i]["utm_zone_letter"],
        )
        point_b = (
            waypoints[i + 1]["utm_x"],
            waypoints[i + 1]["utm_y"],
            waypoints[i + 1]["altitude"],
            waypoints[i + 1]["utm_zone_number"],
            waypoints[i + 1]["utm_zone_letter"],
        )

        j = check_for_collision(point_a, point_b, obstacles, padding)

        if j != None:
            print(
                "Path between waypoints "
                + str(i)
                + " and "
                + str(i + 1)
                + " intersect obstacle "
                + str(j)
            )
            obstacle = (
                obstacles[j]["utm_x"],
                obstacles[j]["utm_y"],
                obstacles[j]["radius"],
                obstacles[j]["height"],
            )
            data = find_new_point(
                point_a,
                point_b,
                obstacle,
                obstacle[2],
                padding,
                max_distance,
                debugging,
            )
            # insert between waypoints
            if len(data) == 2:
                waypoints.insert(i + 1, data[0])
                waypoints.insert(i + 2, data[1])
                num_waypoints += 2  # increase number of waypoints
                i -= 2  # back up two waypoint in case path to new waypoint causes new conflicts
            else:
                waypoints.insert(i + 1, data[0])
                num_waypoints += 1  # increase number of waypoints
                i -= 1  # back up one waypoint in case path to new waypoint causes new conflicts

        i += 1

    return waypoints
