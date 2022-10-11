from typing import Iterable, List, Tuple
from helper import get_bounds
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from cell_map import CellMap
"""
Divides the ODLC search area into identical cells.
"""

CELL_SIZE = 0.00015
CENTER_OFFSET = CELL_SIZE / 2

def segment(polygon: List[Tuple[int, int]]) -> Iterable: #return cell_map
    """
    divides the ODLC search area into a probability map

    Parameters
    ----------
    polygon: List[Tuple[int, int]]
        A list of points defining the polygon
    """
    prob_map_points = []
    bounds = get_bounds(polygon)
    within_check = Polygon(polygon)
    for i in range(math.ceil((bounds['x'][1] - bounds['x'][0]) / CELL_SIZE)):
        row = []
        for j in range(math.ceil((bounds['y'][1] - bounds['y'][0]) / CELL_SIZE)):
            #check if point is within polygon
            x_val = bounds['x'][0] + CENTER_OFFSET + (i * CELL_SIZE)
            y_val = bounds['y'][0] + CENTER_OFFSET + (j * CELL_SIZE)
            p = Point(x_val, y_val)
            if within_check.contains(p):
                row.append((x_val, y_val))
            else:
                row.append('X')
        prob_map_points.append(row)
    
    return prob_map_points

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

    x = segment(test_cloud)
    c = CellMap(x, 100)
    c.display()



    

