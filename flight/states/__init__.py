"""Initialization file for state exporting"""
from typing import Dict
from flight.states.state import State
from flight.states.start_state import Start
from flight.states.pre_processing import PreProcess
from flight.states.takeoff import Takeoff
from flight.states.waypoints import Waypoints
from flight.states.air_drop import AirDrop
from flight.states.standard_odlc import StandardODLC
from flight.states.emergent_odlc import EmergentODLC
from flight.states.offaxis_odlc import OffAxisODLC
from flight.states.land import Land
from flight.states.final_state import Final

STATES: Dict[str, State] = {
    "Start State": Start,
    "Pre-Processing": PreProcess,
    "Takeoff": Takeoff,
    "Waypoints": Waypoints,
    "Air Drop": AirDrop,
    "Standard ODLC": StandardODLC,
    "Emergent ODLC": EmergentODLC,
    "Off Axis ODLC": OffAxisODLC,
    "Land": Land,
    "Final State": Final
}
