import matplotlib.pyplot as plt
from matplotlib import collections as mc
from typing import List, Tuple


Point = Tuple[float, float]


def plot(G, obstacles, boundary, start, goal, path=None) -> None:
    # plot obstacles
    for obstacle in obstacles:
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]
        plt.gca().add_patch(plt.Circle((x, y), radius, color="gray"))

    # plot boundary
    x1, y1 = [], []
    for point in boundary:
        x1.append(point["utm_x"])
        y1.append(point["utm_y"])
    plt.plot(x1, y1, "ko-")

    # plot graph vertices
    px = [p.x for p in G.vertices]
    py = [p.y for p in G.vertices]
    plt.scatter(px, py, c="cyan")
    plt.scatter(G.startpos.x, G.startpos.y, c="black")
    plt.scatter(G.endpos.x, G.endpos.y, c="black")

    # plot graph edges
    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    for line in lines:
        lx = [p.x for p in line]
        ly = [p.y for p in line]
        plt.plot(lx, ly, "go-")

    # plot path
    if path is not None:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        plt.plot(path_x, path_y, "bo-")

    # plot start and end points
    plt.plot(start.x, start.y, c="red", marker="*")
    plt.plot(goal.x, goal.y, c="red", marker="*")

    plt.gca().set_aspect(1)
    plt.show()
