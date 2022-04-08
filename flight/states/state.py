"""Base State class which all other states inherit"""
import logging
from mavsdk import System
from ..state_settings import StateSettings


class State:
    """
    Base State class

    Attributes
    ----------
        drone (System)
            The drone object; used for flight.

    Methods
    -------
        run() -> None
            Performs all movement operations contained within the state
        _check_arm_or_arm() -> None
            Determines if drone is armed, and if not, arms it.

    """

    def __init__(self, state_settings: StateSettings) -> None:
        """
        Initializes base State class using a StateSettings class

        Parameters
        ----------
            state_settings: StateSettings
                class which contains basic competition data

        Returns
        -------
            None
        """
        logging.info('State "%s" has begun', self.name)
        self.state_settings: StateSettings = state_settings

    async def run(self, drone: System) -> None:
        """
        Runs the functions and code present in state for competition goals

        Parameters
        ----------
            drone: System
                drone object for MAVSDK control

        Returns
        -------
            None

        Raises
        ------
            General
                raises if no code is present in run function
        """
        raise Exception("Run not implemented for state")

    async def _check_arm_or_arm(self, drone: System) -> None:
        """
        Verifies that the drone is armed, if not armed, arms the drone

        Parameters
        ----------
            drone: System
                The drone system; used for flight operations.

        Returns
        -------
            None
        """
        async for is_armed in drone.telemetry.armed():
            if not is_armed:
                logging.debug("Not armed. Attempting to arm")
                await drone.action.arm()
            else:
                logging.warning("Drone armed")
                break

    @property
    def name(self):
        """
        Getter function to return the name of the class

        Parameters
        ----------
            N/A

        Returns
        -------
            Name property of current state class
        """
        return type(self).__name__
