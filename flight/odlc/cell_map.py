from typing import Collection, Iterable, List, Dict
from xml.dom import EMPTY_PREFIX
from helper import coord_conversion
# constants to check:
# P(see | height && r)
# Vision radius

#path algo
# greedy, reduce most unseen cells like chess

class CellMap:
    def get_bounds(self, points: Collection[Iterable[int]]) -> List[List]:
        """
        returns the vertices of the smallest square that encompasses all
        the given points.

        Parameters
        ----------
        points: Collection[Iterable[int]]
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

    def __init_map_shape(self, bounds: Dict[chr, Iterable[int]]) -> List[List]:
        """
        returns an empty array with a shape given the bounds.
        """
        return_list = []
        for _ in range(bounds['y'][0], bounds['y'][1] + 1):
            row = []
            for _ in range(bounds['x'][0], bounds['x'][1] + 1):
                row.append(0)
            return_list.append(row)
    
        return return_list

    def __init_map(self, points: Collection[Iterable[int]]) -> List[List]:
        """
        Creates the list of the map and populates it with cells
        """
        empty = self.__init_map_shape(self.get_bounds(points))
        for point in points:
            new_point = coord_conversion(point, (len(empty[0]), len(empty)))
            empty[new_point[0]][new_point[1]] = 1
        

        return empty

    def display(self):
        for i in range(len(self.data)):
            print(self.data[i])

    def __init__(self, points: Collection[Iterable[int]]):
        """
        Parameters
        ----------
        points: Collection[Iterable[int]]
        a collection of x,y coordinates to define the map
        """
        self.data = self.__init_map(points)


if __name__ == "__main__":
    test_points = [
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 1), (1, 1), (2, 1),
        (0, 0), (1, 0), (2, 0),
    ]
    test_map = CellMap(test_points)
    test_map.display()
