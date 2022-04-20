"""
File containing the move_to function responsible
for moving the drone to a certain waypoint
"""

import typing
import asyncio
from mavsdk import System
import logging
import math

async def move_to(drone: System, latitude: float, longitude: float, altitude: float) -> None:
    """
    This function takes in a latitude, longitude and altitude and autonomously
    moves the drone to that waypoint. This function will also auto convert the altitude
    from feet to meters

    Args:
        drone (System): a drone object that has all offboard data needed for computation
        latitude (float): a float containing the requested latittude to move to
        longitude (float): a float containing the requested longitude to move to
        altitude (float): a float contatining the requested altitude to go to (in meters)

    Returns:
        None
    """
    

    #converts feet into meters
    altitude = altitude * .3048

    #get current altitude
    async for terrain_info in drone.telemetry.home():
        absolute_altitude: float = terrain_info.absolute_altitude_m
        break

    await drone.action.goto_location(latitude,longitude, altitude+absolute_altitude, 0)
    location_reached: bool=False

    #Loops until the waypoint is reached
    while(not location_reached):
        print("Going to waypoint")
        async for position in drone.telemetry.position():
            #continuously checks current latitude, longitude and altitude of the drone
            drone_lat: float=position.latitude_deg
            drone_long: float=position.longitude_deg
            drone_alt: float=position.relative_altitude_m

            #checks if location is reached and moves on if so
            if ((round(drone_lat,2)==round(latitude,2)) and 
                (round(drone_long,2)==round(longitude,2)) and 
                (round(drone_alt,2)==round(altitude,2))):
                location_reached=True
                print("arrived")
                break

        await asyncio.sleep(2)
    return

