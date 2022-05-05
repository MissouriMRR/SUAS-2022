"""State to fly towards Off Axis ODLC for image capture"""
import logging
from mavsdk import System
from ..state_settings import StateSettings
from state import State
from states.land import Land


class OffAxisODLC(State):
    """
    State that handles Off Axis ODLC by flying to closest location w/in boundary and capture image

    Attributes
    ----------
        N/A

    Methods
    -------
        run() -> Land
            Flies to closest location within flight boundary and record off-axis ODLC
    """
    async def run(self, drone: System) -> Land:
        """
        Flies to closest point within flight boundary and record image of Off-Axis ODLC

        Parameters
        ----------
            drone: System
                MAVSDK object for drone control

        Returns
        -------
            Land : State
                State to return to takeoff location and land the drone
        """
        return Land(self.state_settings)
