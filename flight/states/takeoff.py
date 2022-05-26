"""Run takeoff function to launch drone to minimum altitude"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings
from waypoints import Waypoints


class Takeoff(State):
    """
    Runs manual takeoff process, to lift drone to 100ft altitude

    Methods
    -------
        run() -> Waypoints
            runs drone movement functions to rise to 100ft

    Attributes
    ----------
        None
    """
    async def run(self, drone: System) -> Waypoints:
        """
        Run offboard functions to have the drone takeoff

        Parameters
        ----------
            drone: System
                MAVSDK drone object for direct drone control

        Returns
        -------
            Waypoints: next state, Waypoints state to perform waypoint flight plan
        """
        return Waypoints(self.state_settings)
