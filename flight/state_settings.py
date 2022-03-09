"""Class to contain setters and getters for settings in various flight states"""

DEFAULT_WAYPOINTS: int = 14
DEFAULT_RUN_TITLE: str = "N/A"
DEFAULT_RUN_DESCRIPTION: str = "N/A"


class StateSettings:
    """
    Initialize settings for state machine
    Functions:
        __init__: Sets preliminary values for SUAS overheads
        enable_simple_takeoff() -> None: Sets if takeoff procedure should be simple for testing
        set_number_waypoints() -> None: Sets the number of waypoints in flight plan
        set_run_title() -> None: Sets name of competition
        set_run_description() -> None: Sets description for competition
    Member Variables:
        num_waypoints: Number of waypoints on flight plan
        run_title: Title of Competition
        run_description: Description for Competition
    """
    def __init__(self):
        """Default constructor results in default settings"""
        self.simple_takeoff = False
        self.num_waypoints = DEFAULT_WAYPOINTS
        self.run_title = DEFAULT_RUN_TITLE
        self.run_description = DEFAULT_RUN_DESCRIPTION

    # ---- Takeoff settings ---- #

    def enable_simple_takeoff(self, simple_takeoff: bool) -> None:
        """
        Setter for whether to perform simple takeoff instead of regular takeoff
            simple_takeoff(bool): True for drone to go straight up, False to behave normally
        """
        self.simple_takeoff = simple_takeoff

    # ---- Waypoint Settings ---- #

    def set_number_waypoints(self, waypoints: int) -> None:
        """Set the number of waypoints given by Interop System"""
        self.num_waypoints = waypoints

    # ---- Other settings ---- #

    def set_run_title(self, title: str) -> None:
        """Set a title for the run/test to be output in logging"""
        self.run_title = title

    def set_run_description(self, description: str) -> None:
        """Set a description for the run/test to be output in logging"""
        self.run_description = description
