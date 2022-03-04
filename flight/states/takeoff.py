"""Run takeoff function to launch drone to minimum altitude"""
import logging
from state import State
from ..state_settings import StateSettings
from waypoints import Waypoints


class Takeoff(State):
    async def run(self, drone):
        """Run offboard functions to have the drone takeoff"""
        return Waypoints(self.state_settings)
