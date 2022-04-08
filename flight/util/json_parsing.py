"""Utility function to parse mission data and retrieve specific component data"""
from typing import Dict, List
import json


def waypoint_parsing(filename: str) -> List[Dict[str, float]]:
    """
    Accepts name of JSON file and extracts waypoint data for SUAS mission

    Parameters
    ---------
    filename: str
        String of data file to open and access waypoint data

    Returns
    -------
    List[Dict[str, float]]
        List of dictionaries containing latitude, longitude and altitude of each waypoint in mission
        Dict[str, float]
                dictionary containing specific coordinate or attribute with string label

    Raises
    ------
    General Exception
        Designed to catch any error that could arise and prevent corruption

    """
    with open(filename) as f:
        try:
            data_set: Dict[str, List] = json.load(f)
        except:
            f.close()
    f.close()

    waypoint_locs: List[Dict[str, float]] = [point for point in data_set["waypoints"]]

    return waypoint_locs


def stationary_obstacle_parsing(filename: str) -> List[Dict[str, float]]:
    """
    Opens passed JSON file and extracts the Stationary obstacle attributes

    Parameters
    ----------
    filename: str
        String of JSON file name and file type

    Returns
    -------
        stationary_obs: List[Dict[str, float]]
            list of dictionaries containing latitude, longitude, radius, and height of obstacles
            Dict[str, float]
                dictionary containing specific coordinate or attribute with string label

    Raises
    ------
    General Exception
        Broad exception to catch any error and prevent corruption

    """
    with open(filename) as f:
        try:
            data_set: Dict[str, List] = json.load(f)
        except:
            f.close()
    f.close()

    stationary_obs: List[Dict[str, float]] = [obs for obs in data_set["stationaryObstacles"]]

    return stationary_obs
