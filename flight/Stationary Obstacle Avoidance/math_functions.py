import math
from typing import List, Tuple


def distance(a: Tuple[int], b: Tuple[int]) -> float:
    """Calculates the euclidean distance between two points

    Args:
        a (Tuple[int]): x,y coordinates for point a
        b (Tuple[int]): x,y coordinates for point b

    Returns:
        float: distance between points a and b
    """
    return abs(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))


# Takes a point and returns the two points that are tangent to a circle of radius r
# src: https://math.stackexchange.com/a/3190374
def tangent_points(point, circle, r):
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


# Returns the equation of a line given two points
# src: https://stackoverflow.com/a/20679579
def line(p1, p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C


# Finds the intersection of two lines
# src: https://stackoverflow.com/a/20679579
def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# Returns a circle with a center and a radius given 3 points on the edge
def find_circle(b, c, d):
    temp = c[0] ** 2 + c[1] ** 2
    bc = (b[0] ** 2 + b[1] ** 2 - temp) / 2
    cd = (temp - d[0] ** 2 - d[1] ** 2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        return None

    # Center of circle
    cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((cx - b[0]) ** 2 + (cy - b[1]) ** 2) ** 0.5

    return ((cx, cy), radius)


# Calculates which pair of tangent points are on the same side of the circle
# returns line segments between the start and end points and the appropriate tangents
def resolve_closest_tangents(start, end, A1, A2, B1, B2):
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
