"""Run takeoff function to launch drone to minimum altitude"""
import logging
from mavsdk import System
from flight.states.state import State
from flight.state_settings import StateSettings
from flight.states.waypoints import Waypoints


class Takeoff(State):
    """
    Runs manual takeoff process, to lift drone to 100ft altitude

    Attributes
    ----------
        None

    Methods
    -------
        run() -> Waypoints
            runs drone movement functions to rise to 100ft
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
            Waypoints
                Waypoint flying state to perform waypoint flight plan
        """
        return Waypoints(self.state_settings)
