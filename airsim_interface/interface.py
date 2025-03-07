"""
handles interface with unreal airsim thing
running this file will run airsim
"""

import time

import airsim, subprocess, os
from airsim import MultirotorClient

from airsim_interface.load_settings import get_settings, Keychain


def engine_started():
    """
    checks if unreal engine is running by trying to connect an airsim client
    """
    client = airsim.MultirotorClient()  # we are using the multirotor client
    try:
        client.confirmConnection()
        return True
    except:
        # failed connection
        return False


def start_game_engine(project=None, open_gui=True, start_paused=True, join=False):
    """
    starts unreal enging
    Args:
        project: what project to open, defaults to Keychain.Defaultproj
        open_gui: whether to open the gui for the project
        start_paused: guess what it means
        join: if true, this function will run until the project is fully open
    Returns:

    """
    sett = get_settings()
    if project is None:
        project = sett[Keychain.Defaultproj]

    cmd = [sett[Keychain.UE4loc], project, '-game', '-windowed']
    if not open_gui:
        cmd += ['-renderoffscreen', '-nosplash', '-nullrhi']
    # os.system(' '.join(cmd))
    thing = subprocess.Popen(cmd)
    if join:
        time.sleep(10)
        while not engine_started():
            time.sleep(10)
        client = connect_client()
        if start_paused:
            client.simPause(True)
    return thing


def connect_client(client=None, vehicle_name=''):
    if client is None:
        client = airsim.MultirotorClient()  # we are using the multirotor client

    try:
        client.confirmConnection()
    except:
        # failed connection
        return None
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    return client


def disconnect_client(client, vehicle_name=''):
    """
    disarms drone and removes api control from client

    """
    client.armDisarm(False, vehicle_name=vehicle_name)
    client.enableApiControl(False, vehicle_name=vehicle_name)


def quick_land(client: MultirotorClient, vehicle_name=''):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 0, 1, vehicle_name=vehicle_name).join()


def quick_takeoff(client, vehicle_name=''):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 1, 1, vehicle_name=vehicle_name).join()


def step(client, seconds=.5, cmd=lambda: None, pause_after=True):
    """
    steps a simulation
    Args:
        seconds: number of seconds to continue for
        cmd: ASYNCRHONOUS command to run for frames
            if cmd is not asyncronous, it will run fully on a paused simulation
    Returns:

    """
    if client.simIsPause():
        client.simPause(False)
    cmd()
    client.simContinueForTime(seconds=seconds)
    if pause_after:
        client.simPause(True)


def move_along_pth(client: MultirotorClient, pth, v=1., vehicle_name=''):
    """
    Args:
        client: client
        pth: path, sequence of (x,y,z)
        v: velocity, either scalar or list (length of pth)
    """
    if type(v) == float or type(v) == int:
        v = [v for _ in pth]
    for t, v in zip(pth, v):
        if len(t) == 2:
            z = client.simGetVehiclePose(vehicle_name=vehicle_name).position.z_val
            x, y = t
        else:
            x, y, z = t
        client.moveToPositionAsync(x=x, y=y, z=z, velocity=v, vehicle_name=vehicle_name).join()


if __name__ == '__main__':
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--headless", action='store_true', required=False,
                        help="open in headless mode")
    PARSER.add_argument("--start-paused", action='store_true', required=False,
                        help="starts simulation paused")
    args = PARSER.parse_args()
    if not engine_started():
        thing = start_game_engine(open_gui=not args.headless,
                                  start_paused=False,
                                  join=True,
                                  )
