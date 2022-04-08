"""Final state to end all drone processes"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings


class Final(State):
    """
    Empty state to end flight

    Attributes
    ----------
        N/A

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

        Returns
        -------
            None
        """
        return
