import matplotlib.pyplot as plt
from matplotlib import collections as mc
from typing import List, Tuple


Point = Tuple[float, float]


def plot(obstacles, boundary, G=None, path=None, ellr=None, informed_boundary=None) -> None:
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
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        plt.plot(path_x, path_y, "bo-")

    # plot informed boundary
    if informed_boundary is not None:
        informed_boundary_coords = list(informed_boundary.exterior.coords)
        x2 = [p[0] for p in informed_boundary_coords]
        y2 = [p[1] for p in informed_boundary_coords]
        plt.plot(x2, y2, "ko-")

    # plot start and end points
    if G is not None:
        plt.plot(G.startpos.x, G.startpos.y, c="red", marker="*")
        plt.plot(G.endpos.x, G.endpos.y, c="red", marker="*")

    plt.gca().set_aspect(1)
    plt.show()
