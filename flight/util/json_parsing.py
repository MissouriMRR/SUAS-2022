import json


def waypoint_parsing(filename: str):
    f = open(filename, )
    data_set = json.load(f)
    # print(data_set)
    f.close()

    waypoint_Locs = []

    for i in range(0, len(data_set["waypoints"])):
        waypoint_Locs.append(data_set["waypoints"][i])

    return waypoint_Locs


def stationary_obstacle_parsing(filename: str):
    f = open(filename, )
    data_set = json.load(f)
    # print(data_set)
    f.close()

    stationary_Obs = []

    for i in range(0, len(data_set["stationaryObstacles"])):
        stationary_Obs.append(data_set["stationaryObstacles"][i])

    return stationary_Obs
