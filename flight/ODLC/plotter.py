import matplotlib.pyplot as plt
from typing import List, Dict


def plot_data(
    odlc: Dict,
    closest_point: Dict,
    old_boundary: List[Dict],
    new_boundary: List[Dict],
    obstacles: List[Dict],
) -> None:
    # plot obstacles
    for obstacle in obstacles:
        x = obstacle["utm_x"]
        y = obstacle["utm_y"]
        radius = obstacle["radius"]

        plt.gca().add_patch(plt.Circle((x, y), radius, color="red"))

    # plot boundary 1
    x1, y1 = [], []
    for point in old_boundary:
        x1.append(point["utm_x"])
        y1.append(point["utm_y"])
    plt.plot(x1, y1, "ro-")

    # plot boundary 2
    x2, y2 = [], []
    for point in new_boundary:
        x2.append(point[0])
        y2.append(point[1])
    plt.plot(x2, y2, "bo-")

    # plot odlc and closest point to odlc
    plt.plot(odlc["utm_x"], odlc["utm_y"], marker="*")
    plt.plot(closest_point["utm_x"], closest_point["utm_y"], marker="*")

    plt.gca().set_aspect(1)
    plt.show()
