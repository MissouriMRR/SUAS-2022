"""
Main driver code for moving drone to each waypoint
"""

import asyncio
import json_parsing
import goto
from mavsdk import System
import mavsdk as sdk
import logging
import math
import typing
from typing import Dict,List

async def run() -> None:
    #Put all latitudes, longitudes and altitudes into seperate arrays
    lats: List[float]=[]
    longs: List[float]=[]
    altitudes: List[float]=[]
    waypoints: List[Dict[str,float]] =json_parsing.waypoint_parsing("numbers.json")
    for i in waypoints:
        for key,val in i.items():
            if(key=="latitude"):
                lats.append(val)
            if(key=="longitude"):
                longs.append(val)
            if(key=="altitude"):
                altitudes.append(val)

    #create a drone object
    drone: System = System()
    await drone.connect(system_address="udp://:14540")

    #connect to the drone
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("Global position estimate ok")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    #wait for drone to take off
    await asyncio.sleep(20)

    #move to each waypoint in mission
    for point in range(len(waypoints)):
        await goto.move_to(drone,lats[point],longs[point],altitudes[point])

    #return home
    print("Last waypoint reached")
    print("Returning to home")
    await drone.action.return_to_launch()
    print("Staying connected, press Ctrl-C to exit")
    
    #infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

