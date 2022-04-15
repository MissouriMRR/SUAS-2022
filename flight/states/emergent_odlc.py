"""State to search for Emergent ODLC object"""
import logging
from mavsdk import System
from ..state_settings import StateSettings
from state import State
from states.offaxis_odlc import OffAxisODLC


class EmergentODLC(State):
    """
    State where drone flies to last known GPS location and searches for Emergent ODLC

    Attributes
    ----------
        N/A

    Methods
    -------
        run() -> Land
            State to fly to last known position and scan for ODLC
    """
    async def run(self, drone: System) -> OffAxisODLC:
        """
        Flies to last known location and searches around location for Emergent ODLC

        Parameters
        ----------
            drone: System
                MAVSDK object for drone control

        Returns
        -------
            Land : State
                State where drone returns to takeoff location and lands
        """
        return OffAxisODLC(self.state_settings)