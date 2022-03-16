from typing import Dict, List
import json


def waypoint_parsing(filename: str) -> List[Dict[str, float]]:
    """Accepts name of JSON file and extracts waypoint data from JSON
    Args:
        filename: str - String of data file to open and access waypoint data
    Returns:
        Dict - dictionary containing latitude, longitude and altitude of each waypoint in mission
    """
    f = open(filename, )
    data_set: Dict[str, List] = json.load(f)
    f.close()

    waypoint_locs: List[Dict[str, float]] = []

    for i in range(0, len(data_set["waypoints"])):
        waypoint_locs.append(data_set["waypoints"][i])

    return waypoint_locs


def stationary_obstacle_parsing(filename: str):
    f = open(filename, )
    data_set = json.load(f)
    # print(data_set)
    f.close()

    stationary_Obs = []

    for i in range(0, len(data_set["stationaryObstacles"])):
        stationary_Obs.append(data_set["stationaryObstacles"][i])

    return stationary_Obs
