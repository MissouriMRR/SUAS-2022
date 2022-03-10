import math
from typing import Tuple


Point = Tuple[float, float]
Line = Tuple[float, float, float]


def distance(a: Point, b: Point) -> float:
    """
    Calculates the euclidean distance between two points

    Args:
        a (Point): First point
        b (Point): Second point

    Returns:
        float: Returns the distance between points a and b
    """
    return abs(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))


def tangent_points(point: Point, circle: Point, r: float) -> Tuple[Point, Point]:
    """
    From a point outside ther circle, calculates the two points that are tangent to the circle

    Args:
        point (Point): a point that lies outside the radius of the circle
        circle (Point): center of the circle
        r (float): radius of the circle

    Returns:
        Tuple[Point, Point]: the coordinates of each tangent to the circle
    """
    Px = point[0]
    Py = point[1]
    Cx = circle[0]
    Cy = circle[1]
    dx, dy = Px - Cx, Py - Cy
    dxr, dyr = -dy, dx
    d = math.sqrt(dx**2 + dy**2)
    rho = r / d
    ad = rho**2
    bd = rho * math.sqrt(1 - rho**2)
    T1x = Cx + ad * dx + bd * dxr
    T1y = Cy + ad * dy + bd * dyr
    T2x = Cx + ad * dx - bd * dxr
    T2y = Cy + ad * dy - bd * dyr
    return ((T1x, T1y), (T2x, T2y))


def line(p1: Point, p2: Point) -> Line:
    """
    Finds the standard form of the equation of a line given two points

    Args:
        p1 (Point): First pair of x,y coordinates
        p2 (Point): Second pair of x,y coordinates

    Returns:
        Line: The real numbers A, B and -C which make up the standard line equation
    """
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C


def intersection(L1: Line, L2: Line) -> Point:
    """
    Finds the intersection of two lines

    Args:
        L1 (Line): equation of a line in standard form
        L2 (Line): equation of a line in standard form

    Returns:
        Point: coordinates of the intersection
    """
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def resolve_closest_tangents(
    start: Point, end: Point, A1: Point, A2: Point, B1: Point, B2: Point
) -> Tuple[Line, Line, Line, Line]:
    """
    Calculates which pair of tangent points are on the same side of the circle

    Args:
        start (Point): starting point
        end (Point): ending point
        A1 (Point): tangent one of the starting point
        A2 (Point): tangent two of the starting point
        B1 (Point): tangent one of the ending point
        B2 (Point): tangent two of the ending point

    Returns:
        Tuple[Line, Line, Line, Line]: Line segments from the start to the tangent to the end
    """
    distances = [distance(A1, B1), distance(A1, B2), distance(A2, B1), distance(A2, B2)]
    i = distances.index(min(distances))

    if i == 0:
        # then A1, B1 are on the same side, A2, B2 on the other
        route1_segment1 = line(start, A1)
        route1_segment2 = line(B1, end)
        route2_segment1 = line(start, A2)
        route2_segment2 = line(B2, end)
    elif i == 1:
        # then A1, B2 are on the same side, A2, B1 on the other
        route1_segment1 = line(start, A1)
        route1_segment2 = line(B2, end)
        route2_segment1 = line(start, A2)
        route2_segment2 = line(B1, end)
    elif i == 2:
        # then A2, B1 are on the same side, A1, B2 on the other
        route1_segment1 = line(start, A2)
        route1_segment2 = line(B1, end)
        route2_segment1 = line(start, A1)
        route2_segment2 = line(B2, end)
    elif i == 3:
        # then A2, B2 are on the same side, A1, B1 on the other
        route1_segment1 = line(start, A2)
        route1_segment2 = line(B2, end)
        route2_segment1 = line(start, A1)
        route2_segment2 = line(B1, end)
    else:
        print("Error: index was " + str(i) + ", expected index range 0-3.")

    return route1_segment1, route1_segment2, route2_segment1, route2_segment2
