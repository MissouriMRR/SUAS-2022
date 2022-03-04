"""Initialization file for state exporting"""
from states.state import State
from states.pre_processing import PreProcess
from states.takeoff import Takeoff
from states.waypoints import Waypoints

STATES = {
    "Pre-Processing": PreProcess,
    "Takeoff": Takeoff,
    "Waypoints": Waypoints
}
