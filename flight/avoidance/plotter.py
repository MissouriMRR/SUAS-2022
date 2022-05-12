"""Provides plotting functionality for visaulizing waypoint, obstacle, and boundary data"""

from typing import List, Dict, Optional
from shapely.geometry import Polygon, Point
from path_finder import Graph  # pylint: disable=cyclic-import
import matplotlib.pyplot as plt


def plot(
    obstacles: List[Dict[str, float]],
    boundary: List[Dict[str, float]],
    graph: Optional[Graph] = None,
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
    graph : Optional[Graph], optional
        A graph of all nodes and vertices used during path finding, by default None
    path : Optional[List[Point]], optional
        A list of nodes that form a path, by default None
    informed_boundary : Optional[Polygon], optional
        The informed area used during path finding, by default None
    """
    # pylint: disable=too-many-locals

    # plot obstacles
    for obstacle in obstacles:
        plt.gca().add_patch(
            plt.Circle((obstacle["utm_x"], obstacle["utm_y"]), obstacle["radius"], color="gray")
        )

    # plot boundary
    boundary_x = [p["utm_x"] for p in boundary]
    boundary_y = [p["utm_y"] for p in boundary]
    plt.plot(boundary_x, boundary_y, "ko-")

    if graph is not None:
        # plot graph vertices
        graph_x = [p.x for p in graph.vertices]
        graph_y = [p.y for p in graph.vertices]
        plt.scatter(graph_x, graph_y, c="cyan")

        # plot graph edges
        lines = [(graph.vertices[edge[0]], graph.vertices[edge[1]]) for edge in graph.edges]
        for line in lines:
            line_x = [p.x for p in line]
            line_y = [p.y for p in line]
            plt.plot(line_x, line_y, "co-")

    # plot path
    if path is not None:
        path_x = [p[0].x for p in path]
        path_y = [p[0].y for p in path]
        # plot all waypoints
        plt.plot(path_x, path_y, "bo-")
        # plot start and end points
        plt.plot(path[0][0].x, path[0][0].y, c="red", marker="*")
        plt.plot(path[len(path) - 1][0].x, path[len(path) - 1][0].y, c="red", marker="*")
        # plot altitude labels beside each waypoint
        for i, _ in enumerate(path):
            plt.text(path[i][0].x, path[i][0].y, str(round(path[i][1])))

    # plot informed boundary
    if informed_boundary is not None:
        informed_boundary_coords = list(informed_boundary.exterior.coords)
        iboundary_x = [p[0] for p in informed_boundary_coords]
        iboundary_y = [p[1] for p in informed_boundary_coords]
        plt.plot(iboundary_x, iboundary_y, "ko-")

    # plot start and end points
    if graph is not None:
        plt.plot(graph.q_start.x, graph.q_start.y, c="red", marker="*")
        plt.plot(graph.q_goal.x, graph.q_goal.y, c="red", marker="*")

    plt.gca().set_aspect(1)
    plt.show()
