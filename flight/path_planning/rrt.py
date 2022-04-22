# Inspired by https://gist.github.com/Fnjn/58e5eaa27a3dc004c3526ea82a92de80

import math
import random
import numpy as np
import shapely
import helpers
import time
import plotter
from typing import Tuple
from shapely.geometry import Point, Polygon, LineString
from collections import deque


STEP_SIZE = 100  # meters
NEIGHBORHOOD = 200  # meters
ITERATIONS = 10000  # max number of iterations before failing to find a path
ITERATIONS_AFTER = 100  # max number of iterations performed in the smaller area

flyZones = [
    {
        "altitudeMin": 100.0,
        "altitudeMax": 750.0,
        "boundaryPoints": [
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
        ],
    }
][0]

waypoints = [
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

obstacles = [
    {"latitude": 38.146689, "longitude": -76.426475, "radius": 150.0, "height": 750.0},
    {"latitude": 38.142914, "longitude": -76.430297, "radius": 300.0, "height": 300.0},
    {"latitude": 38.149504, "longitude": -76.43311, "radius": 100.0, "height": 750.0},
    {"latitude": 38.148711, "longitude": -76.429061, "radius": 300.0, "height": 750.0},
    {"latitude": 38.144203, "longitude": -76.426155, "radius": 50.0, "height": 400.0},
    {"latitude": 38.146003, "longitude": -76.430733, "radius": 225.0, "height": 500.0},
]


def intersects_obstacle(shape, obstacles):
    for obstacle in obstacles:
        if shape.intersection(obstacle):
            return True
    return False


def nearest(G, q_rand, obstacles):
    q_near = None
    q_near_index = None
    min_dist = float("inf")

    for i, q in enumerate(G.vertices):
        edge = LineString([q, q_rand])  # generate line between testing vertex and q_rand
        if intersects_obstacle(edge, obstacles):  # ensure no collisions
            continue

        # find vertex with closest
        dist = q.distance(q_rand)
        if dist < min_dist:
            min_dist = dist
            q_near = q
            q_near_index = i

    return q_near, q_near_index


def new_vertex(q_rand, q_near, STEP_SIZE):
    dirn = np.array(q_rand) - np.array(q_near)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(STEP_SIZE, length)

    q_new = Point(q_near.x + dirn[0], q_near.y + dirn[1])
    return q_new


def in_boundary(boundary, vertex):
    if boundary.contains(vertex):
        return True
    return False


def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {(startpos.x, startpos.y): 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.0}

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[(pos.x, pos.y)] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def randomPosition(self, boundary):
        return get_random_point_in_polygon(boundary)


def rrt(startpos, endpos, boundary, obstacles):
    G = Graph(startpos, endpos)

    for _ in range(ITERATIONS):
        q_rand = G.randomPosition(boundary)
        if intersects_obstacle(q_rand, obstacles):
            continue

        q_near, q_near_index = nearest(G, q_rand, obstacles)
        if q_near is None:
            continue

        q_new = new_vertex(q_rand, q_near, STEP_SIZE)

        q_new_index = G.add_vex(q_new)
        dist = q_new.distance(q_near)
        G.add_edge(q_new_index, q_near_index, dist)

        # finish condition
        # must be within some distance of endpos
        dist = q_new.distance(G.endpos)
        if dist <= STEP_SIZE:
            end_index = G.add_vex(G.endpos)
            G.add_edge(q_new_index, end_index, dist)
            G.success = True
            print("success")
            break
    return G


def RRT_star(startpos, endpos, boundary, obstacles, informed_boundary_set=False):
    G = Graph(startpos, endpos)

    ellr = None
    informed_boundary = None

    counter = 0

    for i in range(ITERATIONS):
        # print(i)
        if informed_boundary_set:
            # print(f'Counter {counter}')
            counter += 1

        if counter >= ITERATIONS_AFTER:
            print(f"Iterated for {counter} additional times in the smaller area")
            break

        q_rand = G.randomPosition(boundary)
        if intersects_obstacle(q_rand, obstacles):
            continue

        q_near, q_near_index = nearest(G, q_rand, obstacles)
        if q_near is None:
            continue

        q_new = new_vertex(q_rand, q_near, STEP_SIZE)

        q_new_index = G.add_vex(q_new)
        dist = q_new.distance(q_near)
        G.add_edge(q_new_index, q_near_index, dist)
        G.distances[q_new_index] = G.distances[q_near_index] + dist

        # update nearby vertices distance if q_new can help
        # make a shorter path
        for vex in G.vertices:
            if vex == q_new:
                continue

            dist = vex.distance(q_new)
            if dist > NEIGHBORHOOD:
                continue

            line = LineString([vex, q_new])
            if intersects_obstacle(line, obstacles):
                continue

            idx = G.vex2idx[(vex.x, vex.y)]
            if G.distances[q_new_index] + dist < G.distances[idx]:
                G.add_edge(idx, q_new_index, dist)
                G.distances[idx] = G.distances[q_new_index] + dist

        dist = q_new.distance(G.endpos)
        if dist <= STEP_SIZE:
            endidx = G.add_vex(G.endpos)
            G.add_edge(q_new_index, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[q_new_index] + dist)
            except:
                G.distances[endidx] = G.distances[q_new_index] + dist

            G.success = True
            # print('success')
            # break

            if not informed_boundary_set:
                print(f"SUCCESS: Found a path after iterating {i} times")

                informed_boundary_set = True

                path = dijkstra(G)  # get path
                ellr = informed_area(startpos, endpos, path)  # find informed area

                informed_boundary = boundary.intersection(ellr)  # intersect with boundary
                boundary = informed_boundary
                print("Updated search area to the informed boundary")

            # print('success')
            # break

    return G, ellr, informed_boundary


def informed_area(q_start, q_goal, path):
    expansion = 0  # initial expansion amount
    expansion_rate = 10  # meters
    buffer = 0  # meters
    last_loop = False

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
            print(f"expanded {i} times to meet goal")
            return ellr

        if ellr.contains(LineString(path)):
            expansion += buffer  # general buffer
            last_loop = True

        expansion += expansion_rate

    return ellr


def dijkstra(G):
    srcIdx = G.vex2idx[(G.startpos.x, G.startpos.y)]
    dstIdx = G.vex2idx[(G.endpos.x, G.endpos.y)]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float("inf") for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

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
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)


def relax_path(path, obstacles):
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


if __name__ == "__main__":
    # Add utm coordinates to all
    boundary = helpers.all_latlon_to_utm(flyZones["boundaryPoints"])
    obstacles = helpers.all_latlon_to_utm(obstacles)
    waypoints = helpers.all_latlon_to_utm(waypoints)

    # Convert silly units to proper units
    obstacles = helpers.all_feet_to_meters(obstacles)

    # Create shapely representations of everything for use in algorithm
    boundary_shape = helpers.coords_to_shape(boundary)
    obstacle_shapes = helpers.circles_to_shape(obstacles)
    waypoints_points = helpers.coords_to_points(waypoints)

    # Magic
    start = waypoints_points[4]
    goal = waypoints_points[5]
    start_time = time.time()
    G, ellr, informed_boundary = RRT_star(start, goal, boundary_shape, obstacle_shapes)
    print(f"rrt runtime = {(time.time()-start_time):.3f}s")

    if G.success:
        path = dijkstra(G)
        path = relax_path(path, obstacle_shapes)
        plotter.plot(obstacles, boundary, G, path, ellr, informed_boundary)
    else:
        print("major error! could not find a path!")
        plotter.plot(obstacles, boundary, G, ellr, informed_boundary)
