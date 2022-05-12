"""Provides plotting functionality for visaulizing waypoint, obstacle, and boundary data"""

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon, Point
from path_finder import Graph


def plot(
    obstacles: List[Dict[str, float]],
    boundary: List[Dict[str, float]],
    G: Optional[Graph] = None,
    path: Optional[List[Point]] = None,
    informed_boundary: Optional[Polygon] = None,
) -> None:
    """Plots boundary, obstacle, and path data

    Parameters
    ----------
    obstacles : List[Dict[str, float]]
        A list of obstacles in dictionary format
    boundary : List[Dict[str, float]]
        A list of points representing a boundary in dictionary format
    G : Optional[Graph], optional
        A graph of all nodes and vertices used during path finding, by default None
    path : Optional[List[Point]], optional
        A list of nodes that form a path, by default None
    informed_boundary : Optional[Polygon], optional
        The informed area used during path finding, by default None
    """

    # plot obstacles
    for obstacle in obstacles:
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]
        plt.gca().add_patch(plt.Circle((x, y), radius, color="gray"))

    # plot boundary
    x1 = [p["utm_x"] for p in boundary]
    y1 = [p["utm_y"] for p in boundary]
    plt.plot(x1, y1, "ko-")

    if G is not None:
        # plot graph vertices
        px = [p.x for p in G.vertices]
        py = [p.y for p in G.vertices]
        plt.scatter(px, py, c="cyan")

        # plot graph edges
        lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
        for line in lines:
            lx = [p.x for p in line]
            ly = [p.y for p in line]
            plt.plot(lx, ly, "co-")

    # plot path
    if path is not None:
        path_x = [p[0].x for p in path]
        path_y = [p[0].y for p in path]
        path_alts = [p[1] for p in path]
        # plot all waypoints
        plt.plot(path_x, path_y, "bo-")
        # plot start and end points
        plt.plot(path[0][0].x, path[0][0].y, c="red", marker="*")
        plt.plot(path[len(path) - 1][0].x, path[len(path) - 1][0].y, c="red", marker="*")
        # plot altitude labels beside each waypoint
        for i in range(len(path)):
            plt.text(path[i][0].x, path[i][0].y, str(round(path[i][1])))

    # plot informed boundary
    if informed_boundary is not None:
        informed_boundary_coords = list(informed_boundary.exterior.coords)
        x2 = [p[0] for p in informed_boundary_coords]
        y2 = [p[1] for p in informed_boundary_coords]
        plt.plot(x2, y2, "ko-")

    # plot start and end points
    if G is not None:
        plt.plot(G.q_start.x, G.q_start.y, c="red", marker="*")
        plt.plot(G.q_goal.x, G.q_goal.y, c="red", marker="*")

    plt.gca().set_aspect(1)
    plt.show()
