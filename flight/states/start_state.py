"""Beginning state of flight operation, proceed to pre-processing"""
import logging
from mavsdk import System
from state import State
from ..state_settings import StateSettings
from states.pre_processing import PreProcess


class Start(State):
    """
    Preliminary state, proceed state machine into pre-processing

    Attributes
    ----------
        N/A

    Methods
    -------
        begin() -> PreProcess
            Beginning function to start state machine, proceeding to pre-processing step

    """
    async def begin(self, drone: System) -> PreProcess:
        """
        Initialization function to start flight state machine

        Parameters
        ----------
            drone: System
                drone object for directv drone control

        Returns
        -------
            PreProcess : State
                data pre-processing state, to advance state machine
        """
        return PreProcess(self.state_settings)