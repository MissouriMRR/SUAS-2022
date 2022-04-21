# Inspired by https://gist.github.com/Fnjn/58e5eaa27a3dc004c3526ea82a92de80

import math
import random
import numpy as np
from typing import Tuple
from shapely.geometry import Point, Polygon, LineString
from collections import deque


STEP_SIZE = 150  # meters
NEIGHBORHOOD = 150  # meters
ITERATIONS = 500


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


def RRT_star(startpos, endpos, boundary, obstacles):
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
    return G


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
