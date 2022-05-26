"""Run flight plan assembled in pre-processing to fly to all waypoints sequentially"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings
from states.land import Land


class Waypoints(State):
    """
    State to fly to each waypoint in mission plan, using Obstacle Avoidance during flight

    Methods
    -------
        run() -> Land: For each waypoint, flies to Latlon & avoids any obstacles

    Attributes
    ----------
        None
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
            Land: landing state to land drone after finishing tasks
        """
        return Land(self.state_settings)
