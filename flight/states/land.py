"""Landing the drone at home position after each other task is finished"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings
from states.final_state import Final


class Land(State):
    """
    State to land the drone safely after finishing other flight & vision tasks

    Attributes
    ----------
        N/A

    Methods
    -------
        run() -> Final
            Running the landing procedure after returning to home
    """
    async def run(self, drone: System) -> Final:
        """
        Run landing function to have drone slowly return to home position

        Parameters
        ----------
            drone: System
                MAVSDK drone object for direct drone control

        Returns
        -------
            Final : State
                final state for ending state machine
        """
        return Final(self.state_settings)
