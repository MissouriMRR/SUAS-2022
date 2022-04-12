"""Create the mission for the UGV to follow after being dropped from Drone"""

import asyncio
from asyncio import Task
from typing import List
from mavsdk import System
from mavsdk.mission import MissionPlan, MissionItem


async def run(ugv_lat: float, ugv_lon: float) -> None:
    """
    Function to create the mission for the UGV, and upload & run on PX4

    Parameters
    ----------
        N/A

    Returns
    -------
        None
            Nothing is returned
    """
    ugv: System = System()
    # ugv_lat: float = 38.146152      # Latitude from Mission 1 test data
    # ugv_lon: float = -76.426396     # Longitude from Mission 1 test data
    await ugv.connect(system_address="udp://14540")
    async for state in ugv.core.connection_state():
        if state.is_connected:
            print("UGV discovered")
            break

    print_mission_progress_task: Task = asyncio.ensure_future(print_mission_progress(ugv))

    running_tasks: List[Task] = [print_mission_progress_task]
    termination_task: Task = asyncio.ensure_future(observe_is_in_air(ugv, running_tasks))

    mission_items: List[MissionItem] = [MissionItem(ugv_lat, ugv_lon, 0, 10, False, float('nan'), float('nan'),
                                                    MissionItem.CameraAction.NONE,
                                                    float('nan'), float('nan'), float('nan'), float('nan'),
                                                    float('nan'))]
    mission_plan: MissionPlan = MissionPlan(mission_items)
    await ugv.mission.set_return_to_launch_after_mission(False)

    print("-- Uploading mission")
    await ugv.mission.upload_mission(mission_plan)

    print("-- Arming")
    await ugv.action.arm()

    print("-- Starting mission")
    await ugv.mission.start_mission()

    await termination_task


async def print_mission_progress(ugv: System) -> None:
    """
    Displays the mission progress to the standard console

    Parameters
    ----------
        ugv: System
            MAVSDK object representing the physical vehicle to control

    Returns
    -------
        None
    """
    async for mission_progress in ugv.mission.mission_progress():
        print(f"Mission progress: "
              f"{mission_progress.current}/"
              f"{mission_progress.total}")


async def observe_is_in_air(ugv: System, running_tasks: List[Task]) -> None:
    """
    Monitors whether the drone is flying or not and returns after landing

    Parameters
    ----------
        ugv: System
            MAVSDK object for the physical movement of the UGV
        running_tasks: List[Task]
            Structure of tasks for UGV to perform

    Returns
    -------
        None

    Raises
    ------
        asyncio.CancelledError
            UGV fails to cancel a task due to a hardware issue
    """

    was_in_air: bool = False
    async for is_in_air in ugv.telemetry.in_air():
        if is_in_air:
            was_in_air = is_in_air

        if was_in_air and not is_in_air:
            for task in running_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await asyncio.get_event_loop().shutdown_asyncgens()
            return
