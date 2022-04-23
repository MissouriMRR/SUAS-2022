"""Class to contain setters and getters for settings in various flight states"""

DEFAULT_WAYPOINTS: int = 14             # Total number of waypoints flown to in Auto. flight of SUAS2022
DEFAULT_RUN_TITLE: str = "N/A"          # Title of flight mission for logging
DEFAULT_RUN_DESCRIPTION: str = "N/A"    # Description of current flight mission for logging


class StateSettings:
    """
    Initialize settings for state machine

    Methods
    -------
        __init__
            Sets preliminary values for SUAS overheads
        @Property
        simple_takeoff() -> bool
            Setter to determine if testing takeoff procedure desired
        num_waypoints() -> int
            Sets the number of waypoints in the competition
        run_title() -> str
            Sets the name of the current flight for logging
        run_description() -> str
            Sets the description of current flight mission

        @Setters
        simple_takeoff() -> None
            Enables access to status of simple_takeoff
        num_waypoints() -> None
            Enables access to number of waypoints in the competition
        run_title() -> None
            Enables access to description of current flight mission
        run_description() -> None
            Enables access to description of current flight mission

    Attributes
    ----------
        __num_waypoints: int
            Number of waypoints on flight plan
        __run_title: str
            Title of Competition
        __run_description: str
            Description for Competition
    """
    def __init__(self, takeoff_bool: bool = False, num_waypoints: int = DEFAULT_WAYPOINTS, waypoints_bool: bool = False,
                 obstacle_avoidance: bool = False,
                 title: str = DEFAULT_RUN_TITLE, description: str = DEFAULT_RUN_DESCRIPTION) -> None:
        """
        Default constructor results in default settings

        Parameters
        ----------
            takeoff_bool: bool
                determines if a simple takeoff procedure should be executed for testing
            num_waypoints: int
                Number of waypoints, extracted from SUAS mission plan
            waypoints_bool: bool
                Determines if waypoint integration test will be run
            obstacle_avoidance: bool
                Determines if obstacle avoidance integration test will be run
            title: str
                Title of current flight mission, for logging purposes
            description: str
                Description of current flight mission, for logging purposes

        Returns
        -------
            None
        """
        self.__simple_takeoff: bool = takeoff_bool
        self.__num_waypoints: int = num_waypoints
        self.__enable_waypoints: bool = waypoints_bool
        self.__enable_obstacle_avoidance: bool = obstacle_avoidance
        self.__run_title: str = title
        self.__run_description: str = description

    # ---- Takeoff settings ---- #
    @property
    def simple_takeoff(self) -> bool:
        """
        Establishes simple_takeoff as a private member variable

        Parameters
        ----------
            N/A

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

        Returns
        -------
            None
        """
        self.__simple_takeoff = simple_takeoff

    # ---- Waypoint Settings ---- #
    @property
    def num_waypoints(self) -> int:
        """
        Establishes num_waypoints as private member variable

        Parameters
        ----------
            N/A

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

        Returns
        -------
            None
        """
        self.__num_waypoints = waypoints

    @enable_waypoints.setter
    def enable_waypoints(self, waypoints_bool: bool) -> None:
        """
        Setter for if Waypoint flight integration test is enabled

        Parameters
        ----------
            waypoints_bool: bool
                Determines if integration test will be active
        """
        self.__enable_waypoints = waypoints_bool

    # ---- Obstacle Avoidance Settings ---- #
    @enable_obstacle_avoidance.setter
    def enable_obstacle_avoidance(self, obs_avoid: bool) -> None:
        """
        Setter for Obstacle Avoidance integration test

        Parameters
        ----------
            obs_avoid: bool
                Determines if obstacle avoidance integration test will be run
        """
        self.__enable_obstacle_avoidance = obs_avoid

    # ---- Other settings ---- #
    @property
    def run_title(self) -> str:
        """
        Set a title for the run/test to be output in logging

        Parameters
        ----------
            N/A

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

        Returns
        -------
            None
        """
        self.__run_title = title

    @property
    def run_description(self) -> str:
        """
        Set a description for the run/test to be output in logging

        Parameters
        ----------
            N/A

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

        Returns
        -------
            None
        """
        self.__run_description = description
