"""State to receive data from interop system and perform preliminary calculations"""
import logging
from state import State
from ..state_settings import StateSettings
from takeoff import Takeoff


class PreProcess(State):
    async def data_retrieval(self, drone):
        """Prelim function to accept data"""
        return Takeoff(self.state_settings)

