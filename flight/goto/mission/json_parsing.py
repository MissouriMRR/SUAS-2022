"""
File used to parse the json file containing the mission waypoints
"""
import json
import typing
from typing import Dict,List

def waypoint_parsing(filename: str) -> List[Dict[str,float]]:
    """
    Parses the json file for all mission-critical waypoints
    Args:
        filename (str): name of the json file

    Returns:
        Waypoint_Locs ([Dict[str,float]]): List of dictionaries containing 
        a string identifier and float for lattitude, longitude and altitude
    """
    f = open(filename, )
    data_set = json.load(f)
    # print(data_set)
    f.close()

    waypoint_Locs: List[Dict[str,float]] = []

    for i in range(0, len(data_set["waypoints"])):
        waypoint_Locs.append(data_set["waypoints"][i])

    return waypoint_Locs

