"""Integration Test for Stationary Obstacle Avoidance"""
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
        state_settings.enable_obstacle_avoidance = True
        state_settings.simple_takeoff = True

        state_settings.run_title = "Obstacle Avoidance Integration Test"
        state_settings.run_description = "Integration test for Stationary Obstacle Avoidance Algorithm"

        flight_manager: FlightManager = FlightManager(state_settings)
        flight_manager.main()
    except:
        logging.exception("Test failed to run.")
