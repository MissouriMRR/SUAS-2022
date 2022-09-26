from typing import Collection, Iterable, List
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

        print(x_bounds, y_bounds)
        return x_bounds, y_bounds

    def create_map(self, points: Collection[Iterable[int]]) -> List[List]:
        bounds = self.get_bounds(points)

        return "deez"

    def __init__(self, points: Collection[Iterable[int]]):
        """
        Parameters
        ----------
        points: Collection[Iterable[int]]
        a collection of x,y coordinates to define the map
        """
        self.data = self.create_map(points)


if __name__ == "__main__":
    test_points = [
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 1), (1, 1), (2, 1),
        (0, 0), (1, 0), (2, 0),
        (0, -1)
    ]
    test_map = CellMap(test_points)
