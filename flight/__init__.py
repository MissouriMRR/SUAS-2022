"""Initialization file for state exporting"""
from typing import Dict
from states.state import State
from states.start_state import Start
from states.pre_processing import PreProcess
from states.takeoff import Takeoff
from states.waypoints import Waypoints
from states.land import Land
from states.final_state import Final
STATES: Dict[str, State]
STATES = {
    "Start State": Start,
    "Pre-Processing": PreProcess,
    "Takeoff": Takeoff,
    "Waypoints": Waypoints,
    "Land": Land,
    "Final State": Final
}
