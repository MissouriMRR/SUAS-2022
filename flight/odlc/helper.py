from typing import Dict, Iterable, Collection, List

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

def get_bounds(points: Collection[Iterable[float | int]]) -> Dict[chr, List[float]]:
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
            try:
                if points[i][dim[0]] < dim[1][0]: #smallest x | y
                    dim[1][0] = points[i][dim[0]]

                elif points[i][dim[0]] > dim[1][1]: #biggest x | y
                    dim[1][1] = points[i][dim[0]]
            except:
                pass

    return {'x': x_bounds, 'y': y_bounds}

def m_to_lat(m: float) -> float:
    """
    Given a length of m meters, convert that to a fraction of a latitude.
    """
    

if __name__ == "__main__":
    test_cloud = [
        (38.31722979755967,-76.5570186342245),
        (38.3160801028265,-76.55731984244503),
        (38.31600059675041,-76.5568902018946),
        (38.31546739500083,-76.5537620127769),
        (38.31470980862425,-76.5493636141453),
        (38.31424154692598,-76.5466276164690),
        (38.31369801280048,-76.5434238005822),
        (38.3131406794544,-76.54011767488228),
        (38.31508631356025,-76.5396286507867),
        (38.31615083692682,-76.5449773879351),
        (38.31734210679102,-76.5446085046679),
        (38.31859044679581,-76.5519329158383),
        (38.3164700703248,-76.55255360208943),
        (38.31722979755967,-76.5570186342245)
    ]
    x = get_bounds(test_cloud)
    print(x['x'][1] - x['x'][0])