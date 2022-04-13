"""
Provides plotting functionality for visualizing coordinate data
"""

from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def plot_data(search_paths) -> None:
    for path in search_paths:
        x, y = [], []
        for point in path:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "ro-")

    plt.gca().set_aspect(1)
    plt.show()
