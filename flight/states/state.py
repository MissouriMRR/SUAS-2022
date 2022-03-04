"""Base State class which all other states inherit"""
import logging
from mavsdk import System
from ..state_settings import StateSettings


class State:
    """
    Base State class
    Attributes:
        drone (System): The drone object; used for flight.
    """

    def __init__(self, state_settings: StateSettings):
        logging.info('State "%s" has begun', self.name)
        self.state_settings = state_settings

    async def run(self, drone: System):
        """
        Does all of the operations for the state
        Parameters
        ----------
        drone : System
            The drone system; used for flight operations.
        Returns
        -------
        State or None
            Either the next State in the machine or None if the machine stops
        Raises
        ------
        Exception
            If this function hasn't been overloaded.
        """
        raise Exception("Run not implemented for state")

    async def _check_arm_or_arm(self, drone: System):
        """
        Verifies that the drone is armed, if not armed, arms the drone
        Parameters
        ----------
        drone : System
            The drone system; used for flight operations.
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
        """Returns the name of the class"""
        return type(self).__name__