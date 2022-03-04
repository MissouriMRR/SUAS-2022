"""Class to contain setters and getters for settings in various flight states"""

DEFAULT_WAYPOINTS: int = 14
DEFAULT_RUN_TITLE: str = "N/A"
DEFAULT_RUN_DESCRIPTION: str = "N/A"


class StateSettings:
    def __init__(self):
        """Default constructor results in default settings"""

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