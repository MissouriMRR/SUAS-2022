"""Separates data from SUAS mission plan into data structures for use in specific files"""
from typing import Dict, List, Any
import json


def waypoint_parsing(filename: str) -> List[Dict[str, float]]:
    """
    Accepts name of JSON file and extracts waypoint data for SUAS mission

    Parameters
    ----------
        filename: str
            String of data file to open and access waypoint data

    Returns
    -------
        waypoint_locs : List[Dict[str, float]]
            List of dictionaries containing latitude, longitude and altitude of each waypoint in mission
    """
    with open(filename) as f:
        data_set: Dict[str, Any] = json.load(f)

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
        stationary_obs : List[Dict[str, float]]
            list of dictionaries containing latitude, longitude, radius, and height of obstacles
    """
    with open(filename) as f:
        data_set: Dict[str, Any] = json.load(f)

    stationary_obs: List[Dict[str, float]] = [obs for obs in data_set["stationaryObstacles"]]

    return stationary_obs
