from typing import Collection, Iterable, List, Dict, Tuple
from cell import Cell
from helper import coord_conversion, get_bounds
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

    def display(self):
        for i in range(len(self.data)):
            row_string = ""
            for j in range(len(self.data[0])):
                if not self.data[i][j].is_valid:
                    row_string += ' '
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
                    self[poi[0]][poi[1]].probability *= seeker.find_prob
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
    pass
