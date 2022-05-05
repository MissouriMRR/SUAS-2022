"""Integration Test for waypoint flying"""
import os
import sys
import logging
from flight_manager import FlightManager
from flight.state_settings import StateSettings

parent_dir = os.path.dirname(os.path.abspath(__file__))
gparent_dir = os.path.dirname(parent_dir)


if __name__ == "__main__":
    try:
        state_settings: StateSettings = StateSettings()
        state_settings.simple_takeoff = True
        state_settings.enable_waypoints = True

        state_settings.run_title = "Waypoint Flight Integration Test"
        state_settings.run_description = "Fly series of waypoints without Obstacle Avoidance"

        flight_manager: FlightManager = FlightManager(state_settings)
        flight_manager.main()

    except:
        logging.exception("Failure to run test")
