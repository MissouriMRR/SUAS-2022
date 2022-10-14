from typing import Collection, Iterable, List, Dict, Tuple
from cell import Cell
from helper import coord_conversion, get_bounds
from seeker import Seeker
from segmenter import segment
# constants to check:
# P(see | height && r)
# Vision radius

#path algo
# greedy, reduce most unseen cells like chess

class CellMap:
    
    def __init_map(self, points: Iterable[Iterable[Iterable[float]]], ODLCs: int):
        for i in range(len(points)):
            for j in range(len(points[0])):
                if points[i][j] != 'X':
                    points[i][j] = Cell(ODLCs / self.n, False, points[i][j][0], points[i][j][1], True)
                else:
                   points[i][j] = Cell(0, False, None, None, False) 
        return points

    def __getitem__(self, index):
        return self.data[index]

    def display(self, drone_pos: Tuple[int] | None = None):
        for i in range(len(self.data)):
            row_string = ""
            for j in range(len(self.data[0])):
                if not self.data[i][j].is_valid:
                    row_string += ' '
                elif drone_pos == (i, j):
                    row_string += 'S'
                else:
                    row_string += 'X'
            print(row_string)

    def update_probs(self, pos: Tuple[int], seeker: object):
        """
        Given a drone at position pos[0], pos[1], this function updates the probabalities
        of each cell to reflect that the seeker has observed them.
        """

        for disp_vec in seeker.view_vecs:
            try:
                poi = (pos[0] + disp_vec[0], pos[1] + disp_vec[1])
                if poi[0] >= 0 and poi[1] >= 0 and poi not in seeker.current_view:
                    self[poi[0]][poi[1]].probability *= 1 - seeker.find_prob
            except:
                pass
        seeker.current_view = seeker.get_in_view(self)

    def __init__(self, points: Collection[Iterable[Iterable[int]]], ODLCs: int = 1):
        """
        Parameters
        ----------
        points: Collection[Iterable[int]]
        a collection of x,y coordinates to define the map
        """
        self.n = len(points) * len(points[0])
        self.data = self.__init_map(points, ODLCs)

        flat_list = []
        for sub_list in points:
            for item in sub_list:
                if item.x != None and item.y != None:
                    flat_list.append((item.x, item.y))
        self.bounds = get_bounds(flat_list)


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
    area = segment(test_cloud)
    cell_map = CellMap(area, 4)
    seeker = Seeker((4, 108), 0.85, 4, cell_map)
    cell_map.display(seeker.pos)