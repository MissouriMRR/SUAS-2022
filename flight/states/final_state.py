"""Final state to end all drone processes"""
import logging
from mavsdk import System
from flight.states.state import State
from flight.state_settings import StateSettings


class Final(State):
    """
    Empty state to end flight

    Methods
    -------
        run() -> None
            Ends the state machine
    """
    async def run(self, drone: System) -> None:
        """
        Do nothing and end

        Parameters
        ----------
            drone: System
                drone object that must be passed to any function with control of the drone
        """
        return
