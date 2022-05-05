import math
import random
import numpy as np
import shapely
import helpers
import time
import plotter
from typing import List, Tuple, Dict, Union, final
from shapely.geometry import Point, Polygon, LineString
from collections import deque


STEP_SIZE: int = 100  # max distance between vertices in graph (meters)
NEIGHBORHOOD_SIZE: int = 200  # search radius around current node for optimizing path (meters)
MAX_ITERATIONS: int = 10000  # max number of iterations before failing to find a path
INFORMED_ITERATIONS: int = 100  # max number of iterations performed in the informed area


class Graph:
    def __init__(self, q_start: Point, q_goal: Point):
        self.q_start = q_start
        self.q_goal = q_goal

        self.vertices = [q_start]
        self.edges = []
        self.success = False

        self.vertex_to_index = {(q_start.x, q_start.y): 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.0}

    def add_vertex(self, q: Point) -> int:
        try:
            index: int = self.vertex_to_index[q]
        except:
            index: int = len(self.vertices)
            self.vertices.append(q)
            self.vertex_to_index[(q.x, q.y)] = index
            self.neighbors[index] = []
        return index

    def add_edge(self, index1: int, index2: int, cost: float) -> None:
        self.edges.append((index1, index2))
        self.neighbors[index1].append((index2, cost))
        self.neighbors[index2].append((index1, cost))

    def random_position(self, boundary: Polygon) -> Point:
        return get_random_point_in_polygon(boundary)


def intersects_obstacle(shape: Union[Polygon, LineString], obstacles: List[Polygon]) -> bool:
    for obstacle in obstacles:
        if shape.intersection(obstacle):
            return True
    return False


def nearest(G: Graph, q_rand: Point, obstacles: List[Polygon]) -> Tuple[Point, int]:
    q_near: Point = None
    q_near_index: int = None
    min_dist: float = float("inf")

    for i, q in enumerate(G.vertices):
        edge: LineString = LineString(
            [q, q_rand]
        )  # generate line between testing vertex and q_rand
        if intersects_obstacle(edge, obstacles):  # ensure no collisions
            continue

        # find vertex with closest
        dist: float = q.distance(q_rand)
        if dist < min_dist:
            min_dist = dist
            q_near = q
            q_near_index = i

    return q_near, q_near_index


def new_vertex(q_rand: Point, q_near: Point, STEP_SIZE: int) -> Point:
    dirn = np.array(q_rand) - np.array(q_near)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(STEP_SIZE, length)

    q_new: Point = Point(q_near.x + dirn[0], q_near.y + dirn[1])
    return q_new


def in_boundary(boundary: Polygon, vertex: Point) -> bool:
    if boundary.contains(vertex):
        return True
    return False


def get_random_point_in_polygon(polygon: Polygon) -> Point:
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p


def rrt_star(
    q_start: Point,
    q_goal: Point,
    boundary: Polygon,
    obstacles: List[Point],
    informed_boundary_set=False,
) -> Tuple[Graph, Polygon]:
    G: Graph = Graph(q_start, q_goal)

    ellr = None
    informed_boundary = None

    counter = 0

    for i in range(MAX_ITERATIONS):
        if informed_boundary_set:
            counter += 1

        if counter >= INFORMED_ITERATIONS:
            print(f"Iterated for {counter} additional times")
            break

        q_rand = G.random_position(boundary)
        if intersects_obstacle(q_rand, obstacles):
            continue

        q_near, q_near_index = nearest(G, q_rand, obstacles)
        if q_near is None:
            continue

        q_new = new_vertex(q_rand, q_near, STEP_SIZE)

        q_new_index = G.add_vertex(q_new)
        dist = q_new.distance(q_near)
        G.add_edge(q_new_index, q_near_index, dist)
        G.distances[q_new_index] = G.distances[q_near_index] + dist

        # update nearby vertices distance if q_new can help
        # make a shorter path
        for vex in G.vertices:
            if vex == q_new:
                continue

            dist = vex.distance(q_new)
            if dist > NEIGHBORHOOD_SIZE:
                continue

            line = LineString([vex, q_new])
            if intersects_obstacle(line, obstacles):
                continue

            idx = G.vertex_to_index[(vex.x, vex.y)]
            if G.distances[q_new_index] + dist < G.distances[idx]:
                G.add_edge(idx, q_new_index, dist)
                G.distances[idx] = G.distances[q_new_index] + dist

        dist = q_new.distance(G.q_goal)
        if dist <= STEP_SIZE:
            endidx = G.add_vertex(G.q_goal)
            G.add_edge(q_new_index, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[q_new_index] + dist)
            except:
                G.distances[endidx] = G.distances[q_new_index] + dist

            G.success = True

            if not informed_boundary_set:
                print(f"SUCCESS: Found a path after iterating {i} times")

                informed_boundary_set = True

                path = get_path(G)  # get path
                ellr = informed_area(q_start, q_goal, path)  # find informed area

                informed_boundary = boundary.intersection(ellr)  # intersect with boundary
                boundary = informed_boundary
                print("Updated search area to the informed boundary")

    return G, informed_boundary


def informed_area(q_start: Point, q_goal: Point, path: List[Point]) -> Polygon:
    expansion: int = 0  # initial expansion amount
    expansion_rate: int = 10  # meters
    buffer: int = 0  # meters
    last_loop: bool = False

    print("Generating informed area...")

    # Loop until the informed area expands enough to cover all points
    i = 0
    while True:
        i += 1
        # 1st elem = center point (x,y) coordinates
        center = ((q_goal.x + q_start.x) / 2, (q_goal.y + q_start.y) / 2)

        # 2nd elem = the two semi-axis values (along x, along y)
        dist = q_start.distance(q_goal)
        x_semi_axis = (dist / 2) + expansion
        y_semi_axis = (dist / 4) + expansion

        # 3rd elem = angle in degrees between x-axis of the Cartesian base
        #            and the corresponding semi-axis
        delta_x = q_goal.x - q_start.x
        delta_y = q_goal.y - q_start.y
        theta = math.degrees(math.atan2(delta_y, delta_x))

        ellipse = (center, (x_semi_axis, y_semi_axis), theta)

        # Let create a circle of radius 1 around center point:
        circ = shapely.geometry.Point(ellipse[0]).buffer(1)

        # Let create the ellipse along x and y:
        ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))

        # Let rotate the ellipse (clockwise, x axis pointing right):
        ellr = shapely.affinity.rotate(ell, ellipse[2])

        if last_loop:
            print(f"Expanded ellipse {i} times to enclose path")
            return ellr

        if ellr.contains(LineString(path)):
            expansion += buffer  # general buffer
            last_loop = True

        expansion += expansion_rate

    return ellr


def get_path(G: Graph) -> List[Point]:
    src_index = G.vertex_to_index[(G.q_start.x, G.q_start.y)]
    dst_index = G.vertex_to_index[(G.q_goal.x, G.q_goal.y)]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float("inf") for node in nodes}
    prev = {node: None for node in nodes}
    dist[src_index] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float("inf"):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dst_index
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)


def relax_path(path: List[Point], obstacles: List[Point]) -> List[Point]:
    if len(path) < 3:
        return path

    # check forwards
    for j in range(2):
        i = 0
        while True:
            if i + 2 > len(path) - 1:
                break
            front_curr = path[i]
            front_next = path[i + 2]
            line = LineString([front_curr, front_next])
            if intersects_obstacle(line, obstacles):
                i += 1
            else:
                del path[i + 1]
        path.reverse()

    return path


def solve(
    data_boundary: List[Dict[str, float]],
    data_obstacles: List[Dict[str, float]],
    data_waypoints: List[Dict[str, float]],
    show_plot: bool = False,
    debug: bool = False,
):
    # Add utm coordinates to all
    boundary: List[Dict[str, float]] = helpers.all_latlon_to_utm(data_boundary)
    obstacles: List[Dict[str, float]] = helpers.all_latlon_to_utm(data_obstacles)
    waypoints: List[Dict[str, float]] = helpers.all_latlon_to_utm(data_waypoints)

    # Get zone data for main zone space
    zone_num, zone_letter = helpers.get_zone_info(boundary)

    # Convert obstacle height and radius from feet to meters
    obstacles = helpers.all_feet_to_meters(obstacles)

    # Create shapely representations of everything for use in algorithm
    boundary_shape: Polygon = helpers.coords_to_shape(boundary)
    obstacle_shapes: List[Point] = helpers.circles_to_shape(obstacles)
    waypoints_points: List[Tuple[Point, float]] = helpers.coords_to_points(waypoints)

    final_route: List[Tuple[float, float, float]] = []

    start_time_final_route = time.time()

    # run rrt on each pair of waypoints
    for i in range(len(waypoints_points) - 1):
        start = waypoints_points[i]
        goal = waypoints_points[i + 1]

        if not intersects_obstacle(LineString([start[0], goal[0]]), obstacle_shapes):
            print(f"Found direct path between waypoints {i} and {i+1}")
            final_route.append(start)
            final_route.append(goal)
            continue

        print(f"Finding path between waypoints {i} and {i+1}")
        start_time = time.time()
        G, informed_boundary = rrt_star(start[0], goal[0], boundary_shape, obstacle_shapes)
        print(f"Solved in {(time.time()-start_time):.3f}s")

        if G.success:
            path = get_path(G)
            path = relax_path(path, obstacle_shapes)
            # Debug plot
            if debug:
                plotter.plot(
                    obstacles, boundary, G=G, path=path, informed_boundary=informed_boundary
                )
            for p in path:
                some_new_altitude = 0
                final_route.append((p, some_new_altitude))
        else:
            print(f"ERROR: Could not find a path after {MAX_ITERATIONS} iterations")

    print(f"Total runtime: {(time.time()-start_time_final_route):.3f}s")

    if show_plot:
        plotter.plot(obstacles, boundary, path=final_route)

    # last step converting back to lat lon
    final_route_latlon: List[Tuple[float, float, float]] = helpers.path_to_latlon(
        final_route, zone_num, zone_letter
    )

    return final_route_latlon


if __name__ == "__main__":
    data_boundary = [
        {"latitude": 38.1462694444444, "longitude": -76.4281638888889},
        {"latitude": 38.151625, "longitude": -76.4286833333333},
        {"latitude": 38.1518888888889, "longitude": -76.4314666666667},
        {"latitude": 38.1505944444444, "longitude": -76.4353611111111},
        {"latitude": 38.1475666666667, "longitude": -76.4323416666667},
        {"latitude": 38.1446666666667, "longitude": -76.4329472222222},
        {"latitude": 38.1432555555556, "longitude": -76.4347666666667},
        {"latitude": 38.1404638888889, "longitude": -76.4326361111111},
        {"latitude": 38.1407194444444, "longitude": -76.4260138888889},
        {"latitude": 38.1437611111111, "longitude": -76.4212055555556},
        {"latitude": 38.1473472222222, "longitude": -76.4232111111111},
        {"latitude": 38.1461305555556, "longitude": -76.4266527777778},
        {"latitude": 38.1462694444444, "longitude": -76.4281638888889},
    ]

    data_waypoints = [
        {"latitude": 38.1446916666667, "longitude": -76.4279944444445, "altitude": 200.0},
        {"latitude": 38.1461944444444, "longitude": -76.4237138888889, "altitude": 300.0},
        {"latitude": 38.1438972222222, "longitude": -76.42255, "altitude": 400.0},
        {"latitude": 38.1417722222222, "longitude": -76.4251083333333, "altitude": 400.0},
        {"latitude": 38.14535, "longitude": -76.428675, "altitude": 300.0},
        {"latitude": 38.1508972222222, "longitude": -76.4292972222222, "altitude": 300.0},
        {"latitude": 38.1514944444444, "longitude": -76.4313833333333, "altitude": 300.0},
        {"latitude": 38.1505333333333, "longitude": -76.434175, "altitude": 300.0},
        {"latitude": 38.1479472222222, "longitude": -76.4316055555556, "altitude": 200.0},
        {"latitude": 38.1443333333333, "longitude": -76.4322888888889, "altitude": 200.0},
        {"latitude": 38.1433166666667, "longitude": -76.4337111111111, "altitude": 300.0},
        {"latitude": 38.1410944444444, "longitude": -76.4321555555556, "altitude": 400.0},
        {"latitude": 38.1415777777778, "longitude": -76.4252472222222, "altitude": 400.0},
        {"latitude": 38.1446083333333, "longitude": -76.4282527777778, "altitude": 200.0},
    ]

    data_obstacles = [
        {"latitude": 38.146689, "longitude": -76.426475, "radius": 150.0, "height": 750.0},
        {"latitude": 38.142914, "longitude": -76.430297, "radius": 300.0, "height": 300.0},
        {"latitude": 38.149504, "longitude": -76.43311, "radius": 100.0, "height": 750.0},
        {"latitude": 38.148711, "longitude": -76.429061, "radius": 300.0, "height": 750.0},
        {"latitude": 38.144203, "longitude": -76.426155, "radius": 50.0, "height": 400.0},
        {"latitude": 38.146003, "longitude": -76.430733, "radius": 225.0, "height": 500.0},
    ]

    route = solve(data_boundary, data_obstacles, data_waypoints, show_plot=True)

    print(route)
