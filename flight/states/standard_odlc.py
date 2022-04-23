"""State to fly in ODLC search grid in set pattern"""
import logging
from mavsdk import System
from flight.state_settings import StateSettings
from flight.states.state import State
from flight.states.land import Land


class StandardODLC(State):
    """
    State to fly ODLC search grid scanning for standard ODLC objects

    Attributes
    ----------
        None

    Methods
    -------
        run() -> Land
            Maneuvers drone in set pattern within ODLC search grid
    """
    async def run(self, drone: System) -> Land:
        """
        Flies drone in ODLC search grid in efficient pattern

        Parameters
        ----------
            drone: System
                MAVSDK object for drone control

        Returns
        -------
            Land
                State in which drone returns to takeoff location and lands
        """
        return Land(self.state_settings)
