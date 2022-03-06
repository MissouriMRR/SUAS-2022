import matplotlib.pyplot as plt
from typing import List, Tuple


Point = Tuple[float, float]


def plot_data(
    waypoints: List[dict],
    obstacles: List[dict],
    padding: float,
    flight_path_color: str = "ko-",
) -> None:
    """
    Plots the waypoints, obstacles, and flight path between waypoints

    Args:
        waypoints (List[dict]): List of dictionaries containing waypoint data
        obstacles (List[dict]): List of dictionaries containing obstacle data
        padding (float): Saftey margin around obstacles
        flight_path_color (str, optional): Color of lines indicating flight path. Defaults to black.
    """

    # plot obstacles
    for obstacle in obstacles:
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]

        plt.gca().add_patch(
            plt.Circle((x, y), radius + padding, color="gray")
        )  # padded obstacle
        plt.gca().add_patch(plt.Circle((x, y), radius, color="red"))  # true obstacle

    # plot waypoints
    x, y = [], []
    for waypoint in waypoints:
        x.append(waypoint["utm_x"])
        y.append(waypoint["utm_y"])
    plt.plot(x, y, flight_path_color)

    plt.gca().set_aspect(1)
    plt.show()


def plot_debug(
    pointA: Point,
    pointB: Point,
    obstacle: Point,
    tangentA1: Point,
    tangentA2: Point,
    tangentB1: Point,
    tangentB2: Point,
    new_point: Point,
    alt_point: Point,
    temp_point: Point,
    extra_1: Point,
    extra_2: Point,
    special_case: bool,
    padding: float,
) -> None:
    """
    Used for debugging. Plots two waypoints and the obstacle between them
    as well as all the other points used in calculations.

    Args:
        pointA (Point): Starting point
        pointB (Point): Ending point
        obstacle (Point): Obstacle
        tangentA1 (Point): First tangent to the starting point
        tangentA2 (Point): Second tangent to the starting point
        tangentB1 (Point): First tangent to the ending point
        tangentB2 (Point): Second tangent to the ending point
        new_point (Point): Intersection point on the shorter side
        alt_point (Point): Intersection point on the longer side
        temp_point (Point): Point that lies on the edge of the circle inline with the center and the new_point
        extra_1 (Point): Additional point if special case is true
        extra_2 (Point): Additional point if special case is true
        special_case (bool): Whether or not the new point is too far from the circle edge
        padding (float): Saftey margin around the obstacle
    """

    radius = obstacle[2]

    # plot obstacle
    plt.gca().add_patch(
        plt.Circle((obstacle[0], obstacle[1]), radius + padding, color="gray")
    )  # padded obstacle
    plt.gca().add_patch(
        plt.Circle((obstacle[0], obstacle[1]), radius, color="red")
    )  # true obstacle

    # prepare point data
    x = [
        pointA[0],
        pointB[0],
        obstacle[0],
        tangentA1[0],
        tangentA2[0],
        tangentB1[0],
        tangentB2[0],
        new_point[0],
        alt_point[0],
        temp_point[0],
        extra_1[0],
        extra_2[0],
    ]
    y = [
        pointA[1],
        pointB[1],
        obstacle[1],
        tangentA1[1],
        tangentA2[1],
        tangentB1[1],
        tangentB2[1],
        new_point[1],
        alt_point[1],
        temp_point[1],
        extra_1[1],
        extra_2[1],
    ]
    col = [
        "green",
        "green",
        "black",
        "black",
        "black",
        "black",
        "black",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
    ]
    labels = [
        "Start",
        "End",
        "Obstacle",
        "A1",
        "A2",
        "B1",
        "B2",
        "New Waypoint",
        "Alt Waypoint",
        "Temp",
        "Extra 1",
        "Extra 2",
    ]

    # plot points
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=col[i])

    # plot text labels
    for i, v in enumerate(labels):
        plt.gca().annotate(v, (x[i], y[i]))

    if special_case:
        plt.plot(
            [pointA[0], extra_1[0], extra_2[0], pointB[0]],
            [pointA[1], extra_1[1], extra_2[1], pointB[1]],
            "go-",
        )
    else:
        plt.plot(
            [pointA[0], new_point[0], pointB[0]],
            [pointA[1], new_point[1], pointB[1]],
            "go-",
        )

    plt.gca().set_aspect(1)
    plt.show()
