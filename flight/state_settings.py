"""Class to contain setters and getters for settings in various flight states"""

DEFAULT_WAYPOINTS: int = 14             # Total number of waypoints flown to in Auto. flight of SUAS2022
DEFAULT_RUN_TITLE: str = "N/A"          # Title of flight mission for logging
DEFAULT_RUN_DESCRIPTION: str = "N/A"    # Description of current flight mission for logging


class StateSettings:
    """
    Initialize settings for state machine
    Functions:
        __init__: Sets preliminary values for SUAS overheads
        simple_takeoff() -> None: Enables access to status of simple_takeoff
                         -> bool: Setter to determine if testing takeoff procedure desired
        num_waypoints() -> None: Enables access to number of waypoints in the competition
                        -> int: Sets the number of waypoints in the competition
        run_title() -> None: Enables access to set name of the current flight
                    -> str: Sets the name of the current flight for logging
        run_description() -> None: Enables access to description of current flight mission
                          -> str: Sets the description of current flight mission
    Member Variables:
        __num_waypoints: Number of waypoints on flight plan
        __run_title: Title of Competition
        __run_description: Description for Competition
    """
    def __init__(self, takeoff_bool: bool = False, num_waypoints: int = DEFAULT_WAYPOINTS,
                 title: str = DEFAULT_RUN_TITLE, description: str = DEFAULT_RUN_DESCRIPTION) -> None:
        """Default constructor results in default settings
        Args:
            takeoff_bool: bool - determines if a simple takeoff procedure should be executed for testing
            num_waypoints: int - Number of waypoints, extracted from SUAS mission plan
            title: str - Title of current flight mission, for logging purposes
            description: str - Description of current flight mission, for logging purposes
        Returns:
            None
        """
        self.__simple_takeoff: bool = takeoff_bool
        self.__num_waypoints: int = num_waypoints
        self.__run_title: str = title
        self.__run_description: str = description

    # ---- Takeoff settings ---- #
    @property
    def simple_takeoff(self) -> bool:
        """Establishes simple_takeoff as a private member variable
        Args:
            N/A
        Returns:
            bool: Status of simple_takeoff variable
        """
        return self.__simple_takeoff

    @simple_takeoff.setter
    def simple_takeoff(self, simple_takeoff: bool) -> None:
        """Setter for whether to perform simple takeoff instead of regular takeoff
        Args:
            simple_takeoff: bool - True for drone to go straight up, False to behave normally
        Returns:
            None
        """
        self.__simple_takeoff = simple_takeoff

    # ---- Waypoint Settings ---- #
    @property
    def num_waypoints(self) -> int:
        """Establishes num_waypoints as private member variable
        Args:
            N/A
        Returns:
            int: Number of waypoints in the competition
        """
        return self.__num_waypoints

    @num_waypoints.setter
    def num_waypoints(self, waypoints: int) -> None:
        """Set the number of waypoints given by Interop System
        Args:
            waypoints: int - Number of waypoints present in mission plan
        Returns:
            None
        """
        self.__num_waypoints = waypoints

    # ---- Other settings ---- #
    @property
    def run_title(self) -> str:
        """Set a title for the run/test to be output in logging
        Args:
            N/A
        Returns:
            str: Created title for flight mission
        """
        return self.__run_title

    @run_title.setter
    def run_title(self, title: str) -> None:
        """Sets the title of the flight mission for logging
        Args:
            title: str - Desired title for the flight
        Returns:
            None
        """
        self.__run_title = title

    @property
    def run_description(self) -> str:
        """Set a description for the run/test to be output in logging
        Args:
            N/A
        Returns:
            str: Desired description for flight mission
        """
        return self.__run_description

    @run_description.setter
    def run_description(self, description: str) -> None:
        """Sets a description of the flight mission
        Args:
            description: str - Written description for flight mission for logs
        Returns:
            None
        """
        self.__run_description = description
