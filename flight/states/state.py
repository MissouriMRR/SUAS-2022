"""Base State class which all other states inherit"""
import logging
from mavsdk import System
from ..state_settings import StateSettings


class State:
    """
    Base State class
    Functions:
        run() -> (State, None): Runs the code present in each state and returns the next state, or None if at the
                                end of the machine
        _check_arm_or_arm() -> None: Determines if drone is armed, and if not, arms it.
    Member Variables:
        drone (System): The drone object; used for flight.
    """

    def __init__(self, state_settings: StateSettings) -> None:
        """ Initializes base State class using a StateSettings class
        Args:
            state_settings: StateSettings - class which contains basic competition data
        Returns:
            None
        """
        logging.info('State "%s" has begun', self.name)
        self.state_settings: StateSettings = state_settings

    async def run(self, drone: System) -> None:
        """ Runs the functions and code present in state for competition goals
        Args:
            drone: System - drone object for MAVSDK control
        Returns:
            None
        Exception:
            General: raises if no code is present in run function
        """
        raise Exception("Run not implemented for state")

    async def _check_arm_or_arm(self, drone: System) -> None:
        """ Verifies that the drone is armed, if not armed, arms the drone
        Args:
            drone: System - The drone system; used for flight operations.
        Returns:
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
        """ Getter function to return the name of the class
        Args:
            N/A
        Returns:
            Name property of current state class
        """
        return type(self).__name__
