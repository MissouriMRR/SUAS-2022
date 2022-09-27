from typing import Iterable, Collection, List

def coord_conversion(coord: Iterable[int], system_dim: Iterable[int], start: str ="ij") -> Iterable[int]:
    """
    Converts between 'xy' coordinates to 'ij'

    Parameters
    ----------
    coord: Iterable[int]
        the coordinate to be converted
    system_dim: Iterable[int]
        The dimensions of the coordinate system
    start: str
        The coordinate system 'coord' is already in
    """

    return (system_dim[1] - coord[1] - 1, coord[0])

def get_bounds(points: Collection[Iterable[float | int]]) -> List[List]:
    """
    returns the vertices of the smallest square that encompasses all
    the given points.

    Parameters
    ----------
    points: Collection[Iterable[float | int]]
    The collection of points that define the given shape
    """
    x_bounds = [float("inf"), float("-inf")]
    y_bounds = [float("inf"), float("-inf")]

    for i in range(len(points)):
        for dim in ((0, x_bounds), (1, y_bounds)):

            if points[i][dim[0]] < dim[1][0]: #smallest x | y
                dim[1][0] = points[i][dim[0]]

            elif points[i][dim[0]] > dim[1][1]: #biggest x | y
                dim[1][1] = points[i][dim[0]]

    return {'x': x_bounds, 'y': y_bounds}