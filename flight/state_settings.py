"""Class to contain setters and getters for settings in various flight states"""

DEFAULT_WAYPOINTS: int = 14             # Total number of waypoints flown to in Auto. flight of SUAS2022
DEFAULT_RUN_TITLE: str = "N/A"          # Title of flight mission for logging
DEFAULT_RUN_DESCRIPTION: str = "N/A"    # Description of current flight mission for logging


class StateSettings:
    """
    Initialize settings for state machine

    Attributes
    ----------
        num_waypoints: int
            Number of waypoints on flight plan
        run_title: str
            Title of Competition
        run_description: str
            Description for Competition

    Methods
    -------
        __init__
            Sets preliminary values for SUAS overheads
    """
    def __init__(self, takeoff_bool: bool = False, num_waypoints: int = DEFAULT_WAYPOINTS,
                 title: str = DEFAULT_RUN_TITLE, description: str = DEFAULT_RUN_DESCRIPTION) -> None:
        """
        Default constructor results in default settings

        Parameters
        ----------
            takeoff_bool: bool
                determines if a simple takeoff procedure should be executed for testing
            num_waypoints: int
                Number of waypoints, extracted from SUAS mission plan
            title: str
                Title of current flight mission, for logging purposes
            description: str
                Description of current flight mission, for logging purposes
        """
        self.__simple_takeoff: bool = takeoff_bool
        self.__num_waypoints: int = num_waypoints
        self.__run_title: str = title
        self.__run_description: str = description

    # ---- Takeoff settings ---- #
    @property
    def simple_takeoff(self) -> bool:
        """
        Establishes simple_takeoff as a private member variable

        Returns
        -------
            bool
                Status of simple_takeoff variable
        """
        return self.__simple_takeoff

    @simple_takeoff.setter
    def simple_takeoff(self, simple_takeoff: bool) -> None:
        """
        Setter for whether to perform simple takeoff instead of regular takeoff

        Parameters
        ----------
            simple_takeoff: bool
                True for drone to go straight up, False to behave normally
        """
        self.__simple_takeoff = simple_takeoff

    # ---- Waypoint Settings ---- #
    @property
    def num_waypoints(self) -> int:
        """
        Establishes num_waypoints as private member variable

        Returns
        -------
            int
                Number of waypoints in the competition
        """
        return self.__num_waypoints

    @num_waypoints.setter
    def num_waypoints(self, waypoints: int) -> None:
        """
        Set the number of waypoints given by Interop System

        Parameters
        ----------
            waypoints: int
                Number of waypoints present in mission plan
        """
        self.__num_waypoints = waypoints

    # ---- Other settings ---- #
    @property
    def run_title(self) -> str:
        """
        Set a title for the run/test to be output in logging

        Returns
        -------
            str
                Created title for flight mission
        """
        return self.__run_title

    @run_title.setter
    def run_title(self, title: str) -> None:
        """
        Sets the title of the flight mission for logging

        Parameters
        ----------
            title: str
                Desired title for the flight
        """
        self.__run_title = title

    @property
    def run_description(self) -> str:
        """
        Set a description for the run/test to be output in logging

        Returns
        -------
            str
                Desired description for flight mission
        """
        return self.__run_description

    @run_description.setter
    def run_description(self, description: str) -> None:
        """
        Sets a description of the flight mission

        Parameters
        ----------
            description: str
                Written description for flight mission for logs
        """
        self.__run_description = description
