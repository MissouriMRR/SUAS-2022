"""Provides path finding functionality using the Informed RRT* algorithm"""

import math
import random
import time
from typing import List, Dict, Tuple, Union, Optional
from collections import deque
from shapely.geometry import Point, Polygon, LineString
import shapely
import numpy as np
import helpers
import plotter


# Global constants
OBSTACLE_BUFFER: int = 5  # buffer obstacles for gps inacuracies (meters)
STEP_SIZE: int = 100  # max distance between vertices in graph (meters)
NEIGHBORHOOD_SIZE: int = 200  # search radius around current node for optimizing path (meters)
MAX_ITERATIONS: int = 10000  # max number of iterations before failing to find a path
INFORMED_ITERATIONS: int = 100  # max number of iterations performed in the informed area


class Graph:
    """A class used to store vertices and edges that make up a graph"""

    def __init__(self, q_start: Point, q_goal: Point):
        self.q_start = q_start
        self.q_goal = q_goal

        self.vertices = [q_start]
        self.edges: List[Tuple[int, int]] = []
        self.success = False

        self.vertex_to_index: Dict[Tuple[float, float], int] = {(q_start.x, q_start.y): 0}
        self.neighbors: Dict[int, List[Tuple[int, float]]] = {0: []}
        self.distances: Dict[int, float] = {0: 0.0}

    def add_vertex(self, q_new: Point) -> int:
        """Adds a vertex to the graph

        Parameters
        ----------
        q_new : Point
            A shapley point

        Returns
        -------
        int
            index of the new vertex in the graph
        """

        try:
            index = self.vertex_to_index[q_new]
        except:  # pylint: disable=bare-except
            index = len(self.vertices)
            self.vertices.append(q_new)
            self.vertex_to_index[(q_new.x, q_new.y)] = index
            self.neighbors[index] = []
        return index

    def add_edge(self, index1: int, index2: int, cost: float) -> None:
        """Adds an edge to the graph between two vertices by their index

        Parameters
        ----------
        index1 : int
            Index of vertex 1
        index2 : int
            Index of vertex 2
        cost : float
            Length of the edge
        """

        self.edges.append((index1, index2))
        self.neighbors[index1].append((index2, cost))
        self.neighbors[index2].append((index1, cost))


def random_position(boundary: Polygon) -> Point:
    """Finds a random point within the boundary

    Parameters
    ----------
    boundary : Polygon
        A closed, bounding shapely polygon

    Returns
    -------
    Point
        A shapely point at a random location inside the boundary
    """

    return get_random_point_in_polygon(boundary)


def intersects_obstacle(shape: Union[Polygon, LineString], obstacles: List[Polygon]) -> bool:
    """Tests whether a shape intersects an obstacle

    Parameters
    ----------
    shape : Union[Polygon, LineString]
        A shapely polygon or line
    obstacles : List[Polygon]
        A list of all the obstacles as shapely shapes

    Returns
    -------
    bool
        Whether or not an intersection was detected
    """

    for obstacle in obstacles:
        if shape.intersection(obstacle):
            return True
    return False


def nearest(
    graph: Graph, q_rand: Point, obstacles: List[Polygon]
) -> Tuple[Optional[Point], Optional[int]]:
    """Finds the nearest vertex in the graph to the given point

    Parameters
    ----------
    graph : Graph
        A graph with at least one vertex
    q_rand : Point
        A shapely point at a random location
    obstacles : List[Polygon]
        A list of all the obstacles as shapely shapes

    Returns
    -------
    Tuple[Point, int]
        The nearest vertex in the graph to q_rand, along with its index
    """

    q_near: Optional[Point] = None
    q_near_index: Optional[int] = None
    min_dist: float = float("inf")

    for i, vertex in enumerate(graph.vertices):
        edge: LineString = LineString(
            [vertex, q_rand]
        )  # generate line between testing vertex and q_rand
        if intersects_obstacle(edge, obstacles):  # ensure no collisions
            continue

        # find vertex with closest
        dist: float = vertex.distance(q_rand)
        if dist < min_dist:
            min_dist = dist
            q_near = vertex
            q_near_index = i

    return q_near, q_near_index


def new_vertex(q_rand: Point, q_near: Point) -> Point:
    """Finds a new vertex to be added to the graph. This vertex is at most
    a distance of STEP_SIZE away from q_near and in the direction of q_rand

    Parameters
    ----------
    q_rand : Point
        A random shapely point located inside a bounding area
    q_near : Point
        The nearest vertex on the graph to q_rand

    Returns
    -------
    Point
        A new shapely point to be added to the graph
    """

    # use numpy to calculate new point along vector
    dirn = np.array(q_rand) - np.array(q_near)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(STEP_SIZE, length)  # type: ignore

    q_new: Point = Point(q_near.x + dirn[0], q_near.y + dirn[1])
    return q_new


def in_boundary(boundary: Polygon, vertex: Point) -> bool:
    """Tests if a point lies within a boundary

    Parameters
    ----------
    boundary : Polygon
        A boundary represented as a shapely polygon
    vertex : Point
        A vertex represented as a shapely point

    Returns
    -------
    bool
        Whether or not the point lies inside the boundary
    """

    if boundary.contains(vertex):
        return True
    return False


def get_random_point_in_polygon(polygon: Polygon) -> Point:
    """Gets a random point inside an oddly shaped polygon by finding the
    minimum bounding rectangle and generating random x,y coordinates until
    a pair is generated that lies within the original polygon

    Parameters
    ----------
    polygon : Polygon
        The desired area for a randomly generated point to lie in

    Returns
    -------
    Point
        A random point guranteed to be inside the polygon
    """

    minx, miny, maxx, maxy = polygon.bounds
    while True:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point):
            return point


def rrt_star(
    q_start: Point, q_goal: Point, boundary: Polygon, obstacles: List[Point]
) -> Tuple[Graph, Polygon]:
    """An implementation of the informed RRT* algorithm https://ieeexplore.ieee.org/document/6942976
    This algorithm generates random points in a contained area to rapidly build out a graph of nodes
    and edges that can be used to find an unobstructed path between a start and a goal point.

    Parameters
    ----------
    q_start : Point
        Starting point
    q_goal : Point
        Goal point
    boundary : Polygon
        A bounding area to restrict the path finding to
    obstacles : List[Point]
        A list of obstacles to avoid

    Returns
    -------
    Tuple[Graph, Polygon]
        A graph of connected vertices that include a path to the goal point
        A polygon representing the informed search area
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    graph: Graph = Graph(q_start, q_goal)

    ellr: Polygon = None
    informed_boundary: Polygon = None
    informed_boundary_set = False

    counter: int = 0

    # begin looping for a maximum number of iterations
    for i in range(MAX_ITERATIONS):
        if informed_boundary_set:
            counter += 1

        if counter >= INFORMED_ITERATIONS:
            print(f"Iterated for {counter} additional times")
            break

        # generate a random point and test for collisions
        q_rand = random_position(boundary)
        if intersects_obstacle(q_rand, obstacles):
            continue

        # find nearest point in graph
        q_near, q_near_index = nearest(graph, q_rand, obstacles)
        if q_near is None or q_near_index is None:
            continue

        # generate a new vertex to be connected to q_near in the direction of q_rand
        q_new = new_vertex(q_rand, q_near)

        q_new_index = graph.add_vertex(q_new)
        dist = q_new.distance(q_near)
        graph.add_edge(q_new_index, q_near_index, dist)
        graph.distances[q_new_index] = graph.distances[q_near_index] + dist

        # update nearby vertices distance if q_new can help make a shorter path
        for vex in graph.vertices:
            if vex == q_new:
                continue

            dist = vex.distance(q_new)
            if dist > NEIGHBORHOOD_SIZE:
                continue

            line = LineString([vex, q_new])
            if intersects_obstacle(line, obstacles):
                continue

            idx = graph.vertex_to_index[(vex.x, vex.y)]
            if graph.distances[q_new_index] + dist < graph.distances[idx]:
                graph.add_edge(idx, q_new_index, dist)
                graph.distances[idx] = graph.distances[q_new_index] + dist

        # test if we are close enough to the goal and can connect a path
        dist = q_new.distance(graph.q_goal)
        if dist <= STEP_SIZE:
            endidx = graph.add_vertex(graph.q_goal)
            graph.add_edge(q_new_index, endidx, dist)
            try:
                graph.distances[endidx] = min(
                    graph.distances[endidx], graph.distances[q_new_index] + dist
                )
            except:  # pylint: disable=bare-except
                graph.distances[endidx] = graph.distances[q_new_index] + dist

            # the first time a path from start to goal has been found
            graph.success = True

            # create an informed search area, update the boundary to this area,
            # and keep generating more nodes in an attempt to improve path
            if not informed_boundary_set:
                print(f"SUCCESS: Found a path after iterating {i} times")

                informed_boundary_set = True

                path = get_path(graph)  # get path
                ellr = informed_area(q_start, q_goal, path)  # find informed area

                informed_boundary = boundary.intersection(ellr)  # intersect with boundary
                boundary = informed_boundary
                print("Updated search area to the informed boundary")

    return graph, informed_boundary


def informed_area(q_start: Point, q_goal: Point, path: List[Point]) -> Polygon:
    """Generates an oval shaped informed area of fixed proportions, scaled to fully enclose
    the path between the start and end points

    Parameters
    ----------
    q_start : Point
        Start point
    q_goal : Point
        Goal point
    path : List[Point]
        A path between the start and goal points

    Returns
    -------
    Polygon
        An oval shaped informed area that fully encloses the path
    """

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


def get_path(graph: Graph) -> List[Point]:
    """Uses Dijkstra's algorithm to find the shortest path between nodes in a graph

    Parameters
    ----------
    graph : Graph
        A graph with nodes and edges

    Returns
    -------
    List[Point]
        The shortest path between the start and goal nodes
    """

    src_index: int = graph.vertex_to_index[(graph.q_start.x, graph.q_start.y)]
    dst_index: int = graph.vertex_to_index[(graph.q_goal.x, graph.q_goal.y)]

    # build dijkstra
    nodes = list(graph.neighbors.keys())
    dist = {node: float("inf") for node in nodes}
    prev = {node: None for node in nodes}
    dist[src_index] = 0

    while nodes:
        cur_node = min(nodes, key=lambda node: dist[node])
        nodes.remove(cur_node)
        if dist[cur_node] == float("inf"):
            break

        for neighbor, cost in graph.neighbors[cur_node]:
            new_cost = dist[cur_node] + cost
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev[neighbor] = cur_node  # type: ignore

    # retrieve path
    path: Point = deque()
    cur_node = dst_index
    while prev[cur_node] is not None:
        path.appendleft(graph.vertices[cur_node])
        cur_node = prev[cur_node]  # type: ignore
    path.appendleft(graph.vertices[cur_node])
    return list(path)


def relax_path(path: List[Point], obstacles: List[Point]) -> List[Point]:
    """Removes as many nodes from a path without causing a collision,
    leading to a straighter path with fewer nodes

    Parameters
    ----------
    path : List[Point]
        A list of nodes
    obstacles : List[Point]
        A list of obstacles represented by shapely shapes

    Returns
    -------
    List[Point]
        An updated path with a reduced number of nodes
    """

    if len(path) < 3:
        return path

    # reduce nodes by moving forwards on the first iteration,
    # then reverse the path and reduce nodes from the other direction
    for _ in range(2):
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


def get_path_length(path: List[Point]) -> float:
    """Gets the total length of a path

    Parameters
    ----------
    path : List[Point]
        A list of nodes

    Returns
    -------
    float
        Total length of the path
    """

    length: float = 0
    for i in range(len(path) - 1):
        length += path[i].distance(path[i + 1])
    return length


def set_altitudes(
    path: List[Point], start_alt: float, goal_alt: float
) -> List[Tuple[Point, float]]:
    """Sets altitudes for every point in a path by calculating the total length
    of a path and assigning a percentage of the altitude difference based on the
    distance each point is along the path

    Parameters
    ----------
    path : List[Point]
        A list of nodes
    start_alt : float
        Altitude at the start node
    goal_alt : float
        Altitude at the goal node

    Returns
    -------
    List[Tuple[Point, float]]
        An updated path where each node is paired with an altitude
    """

    altitude_change = goal_alt - start_alt
    total_length: float = get_path_length(path)
    current_length: float = 0
    path_with_altitudes: List[Tuple[Point, float]] = []

    path_with_altitudes.append((path[0], start_alt))

    for i in range(1, len(path)):
        point = path[i]
        current_length += path[i].distance(path[i - 1])
        fraction_of_alt_change = current_length / total_length
        altitude = start_alt + (altitude_change * fraction_of_alt_change)
        path_with_altitudes.append((point, altitude))

    return path_with_altitudes


def solve(
    d_boundary: List[Dict[str, float]],
    d_obstacles: List[Dict[str, float]],
    d_waypoints: List[Dict[str, float]],
    show_plot: bool = False,
    debug_plot: bool = False,
) -> List[Tuple[float, float, float]]:
    """Calls the informed RRT* algorithm multiple times to solve a path
    between each pair of waypoints in the provided list

    Parameters
    ----------
    d_boundary : List[Dict[str, float]]
        Boundary data in original json format
    d_obstacles : List[Dict[str, float]]
        Obstacle data in original json format
    d_waypoints : List[Dict[str, float]]
        Waypoint data in original json format
    show_plot : bool, optional
        Shows a plot of the complete solution to all the waypoints, by default False
    debug_plot : bool, optional
        Shows a plot of the solution between each pair of waypoints, by default False

    Returns
    -------
    List[Tuple[float, float, float]]
        The solved path as a list of tuples in the format of (lat, long, altitude)
    """
    # pylint: disable=too-many-locals

    # Add utm coordinates to all
    boundary: List[Dict[str, float]] = helpers.all_latlon_to_utm(d_boundary)
    obstacles: List[Dict[str, float]] = helpers.all_latlon_to_utm(d_obstacles)
    waypoints: List[Dict[str, float]] = helpers.all_latlon_to_utm(d_waypoints)

    # Get zone data for main zone space
    zone_num, zone_letter = helpers.get_zone_info(boundary)

    # Convert obstacle height and radius from feet to meters
    obstacles = helpers.all_feet_to_meters(obstacles, OBSTACLE_BUFFER)

    # Create shapely representations of everything for use in algorithm
    boundary_shape: Polygon = helpers.coords_to_shape(boundary)
    obstacle_shapes: List[Point] = helpers.circles_to_shape(obstacles)
    waypoints_points: List[Tuple[Point, float]] = helpers.coords_to_points(waypoints)

    final_route: List[Tuple[Point, float]] = []

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
        graph, informed_boundary = rrt_star(start[0], goal[0], boundary_shape, obstacle_shapes)
        print(f"Solved in {(time.time()-start_time):.3f}s")

        if graph.success:
            path: List[Point] = get_path(graph)
            path = relax_path(path, obstacle_shapes)

            # Plot individual waypoint/obstacle scenarios
            if debug_plot:
                plotter.plot(
                    obstacles, boundary, graph=graph, path=path, informed_boundary=informed_boundary
                )

            path_with_altitudes = set_altitudes(path, start[1], goal[1])
            for j in range(len(path_with_altitudes) - 1):
                final_route.append(path_with_altitudes[j])
        else:
            print(f"ERROR: Could not find a path after {MAX_ITERATIONS} iterations")

    print(f"Total runtime: {(time.time()-start_time_final_route):.3f}s")

    # remove waypoint overlap
    final_route = [
        final_route[i]
        for i in range(len(final_route))
        if (i == 0) or final_route[i] != final_route[i - 1]
    ]

    print(f"{len(final_route)} waypoints")

    if show_plot:
        plotter.plot(obstacles, boundary, path=final_route)

    # last step converting back to lat lon
    final_route_latlon: List[Tuple[float, float, float]] = helpers.path_to_latlon(
        final_route, zone_num, zone_letter
    )

    return final_route_latlon


if __name__ == "__main__":
    data_boundary: List[Dict[str, float]] = [
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

    data_waypoints: List[Dict[str, float]] = [
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

    data_obstacles: List[Dict[str, float]] = [
        {"latitude": 38.146689, "longitude": -76.426475, "radius": 150.0, "height": 750.0},
        {"latitude": 38.142914, "longitude": -76.430297, "radius": 300.0, "height": 300.0},
        {"latitude": 38.149504, "longitude": -76.43311, "radius": 100.0, "height": 750.0},
        {"latitude": 38.148711, "longitude": -76.429061, "radius": 300.0, "height": 750.0},
        {"latitude": 38.144203, "longitude": -76.426155, "radius": 50.0, "height": 400.0},
        {"latitude": 38.146003, "longitude": -76.430733, "radius": 225.0, "height": 500.0},
    ]

    route = solve(data_boundary, data_obstacles, data_waypoints, show_plot=True)

    print(route)
