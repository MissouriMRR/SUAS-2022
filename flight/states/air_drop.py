"""State to maneuver the drone to Air Drop location and release UGV"""
import logging
from mavsdk import System
from ..state_settings import StateSettings
from state import State
from states.land import Land


class AirDrop(State):
    """
    State to move drone to Air Drop GPS location and drop UGV

    Attributes
    ----------
        N/A

    Methods
    -------
        run() -> Land
            Moves drone to Air Drop GPS location and releases UGV for mission
    """
    async def run(self, drone: System) -> Land:
        """
        Move drone to Air Drop GPS and release UGV for ground mission

        Parameters
        ----------
            drone: System
                MAVSDK object for drone control

        Returns
        -------
            Land : State
                State in which drone returns to takeoff location and lands
        """
        return Land(self.state_settings)
