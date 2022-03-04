"""Run flight plan assembled in pre-rpocessing to fly to all waypoints sequentially"""
import logging
from state import State
from ..state_settings import StateSettings


class Waypoints(State):
    async def run(self, drone):
        """Run waypoint path plan with obstacle avoidance running throughout"""
        return
