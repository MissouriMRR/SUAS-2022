"""
Provides plotting functionality for visualizing coordinate data
"""

from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_data(search_paths: List[List[Tuple[float, float]]]) -> None:
    """Simple plotter function to plot the search paths

    Parameters
    ----------
    search_paths : List[Tuple[float, float]]
        A list of search paths to be plotted
    """

    for path in search_paths:
        x, y = [], []
        for point in path:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "ro-")

    plt.gca().set_aspect(1)
    plt.show()
