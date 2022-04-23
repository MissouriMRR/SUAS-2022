"""Run flight plan assembled in pre-processing to fly to all waypoints sequentially"""
import logging
from mavsdk import System
from flight.states.state import State
from flight.state_settings import StateSettings
from flight.states.land import Land


class Waypoints(State):
    """
    State to fly to each waypoint in mission plan, using Obstacle Avoidance during flight

    Attributes
    ----------
        None

    Methods
    -------
        run() -> Land: For each waypoint, flies to Latlon & avoids any obstacles
    """
    async def run(self, drone: System) -> Land:
        """
        Run waypoint path plan with obstacle avoidance running throughout

        Parameters
        ----------
            drone: System
                MAVSDK drone object for direct drone control

        Returns
        -------
            Land
                landing state to land drone after finishing tasks
        """
        return Land(self.state_settings)
