"""State to receive data from interop system and perform preliminary calculations"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings
from takeoff import Takeoff


class PreProcess(State):
    """
    State to accept data and plan out optimal mission structure based on relative distances

    Attributes
    ----------
        N/A

    Methods
    -------
        data_retrieval() -> Takeoff
            Connects to Interop and downloads JSON data from system

    """
    async def data_retrieval(self, drone: System) -> Takeoff:
        """
        Prelim function to accept data

        Parameters
        ----------
            drone: System
                MAVSDK drone object for direct drone control

        Returns
        -------
            Takeoff : State
                the next state, the Takeoff state to advance state machine
        """
        return Takeoff(self.state_settings)
