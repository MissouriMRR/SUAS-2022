import json
import typing

def waypoint_parsing(filename: str):
    f = open(filename, )
    data_set = json.load(f)
    # print(data_set)
    f.close()

    waypoint_Locs: [Dict[str,float]] = []

    for i in range(0, len(data_set["waypoints"])):
        waypoint_Locs.append(data_set["waypoints"][i])

    return waypoint_Locs

