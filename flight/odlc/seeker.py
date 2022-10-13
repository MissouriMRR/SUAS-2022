from typing import Tuple, List
import math
class Seeker:
    def __init__(self, start: Tuple[int], find_prob: float, view: int, prob_map: object) -> object:
        self.index = start
        self.find_prob = find_prob #probability of finding object when cell is in view
        self.view = view
        self.current_view = set([]) # a set of points that the drone is currently looking at
        self.prob_map = prob_map

    def get_view_vecs(self) -> List[Tuple[int]]:
        """
        returns a list of displacement vectors that can be seen from the drone
        based upon its range.
        """
        dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])** 2 + (p1[1] - p2[1])**2)
        view_list = []

        for i in range(self.view * 2 + 1):
            for j in range(self.view * 2 + 1):
                if dist((i, j), (self.view, self.view)) <= self.view:
                    view_list.append((i - self.view, j - self.view))
        return view_list
        

    def move(self, disp_vec: Tuple[int]) -> None:
        try:
            new_pos = [self.index[0] + disp_vec[0], self.index[1] + disp_vec[1]]

            if self.prob_map[new_pos[0]][new_pos[1]].is_valid:
                self.prob_map.update_probs(new_pos, self)
            else:
                pass
        except:
            pass 


if __name__ == "__main__":
    s = Seeker((0, 0), 0.9, 5, [])
    space = s.get_in_view()
    for cell in space:
        print(cell)