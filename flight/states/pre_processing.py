"""State to receive data from interop system and perform preliminary calculations"""
import logging
from mavsdk import System
from flight.states.state import State
from flight.state_settings import StateSettings
from flight.states.takeoff import Takeoff


class PreProcess(State):
    """
    State to accept data and plan out optimal mission structure based on relative distances

    Attributes
    ----------
        None

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
            Takeoff
                The next state, the Takeoff state to advance state machine
        """
        return Takeoff(self.state_settings)
