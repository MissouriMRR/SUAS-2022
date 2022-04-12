"""Utility file to parse different data elements from SUAS mission plan"""
from typing import Dict, List, Tuple
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
        List[Dict[str, float]]
            List of dictionaries containing latitude, longitude and altitude of each waypoint in mission
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
        List[Dict[str, float]]
            list of dictionaries containing latitude, longitude, radius, and height of obstacles
    """
    with open(filename) as f:
        try:
            data_set: Dict[str, List] = json.load(f)
        except:
            f.close()
    f.close()

    stationary_obs: List[Dict[str, float]] = [obs for obs in data_set["stationaryObstacles"]]

    return stationary_obs


def ugv_parsing(filename: str) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Reads the SUAS mission plan and grabs GPS locations for Air Drop

    Parameters
    ----------
        filename: str
            Name of JSON file containing mission data

    Returns
    -------
        Tuple[List[Dict[str, float]], List[Dict[str, float]]]
            Tuple containing GPS coordinates for drone and UGV
    """
    with open(filename) as f:
        try:
            data_set: Dict[str, List] = json.load(f)
        except:
            f.close()
    f.close()

    drop_position: List[Dict[str, float]] = data_set["airDropPos"]
    ugv_destination: List[Dict[str, float]] = data_set["ugvDrivePos"]

    return drop_position, ugv_destination
