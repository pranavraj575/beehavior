import time

import airsim, subprocess, os
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


def start_game_engine(project=None, open_gui=True, start_paused=True, ):
    """
    starts unreal enging
    Args:
        project: what project to open, defaults to Keychain.Defaultproj
        open_gui: whether to open the gui for the project
        start_paused: guess what it means
    Returns:

    """
    sett = get_settings()
    if project is None:
        project = sett[Keychain.Defaultproj]

    cmd = [sett[Keychain.UE4loc], project, '-game', '-windowed']
    if not open_gui:
        cmd += ['-renderoffscreen', '-nosplash', '-nullrhi']
    # os.system(' '.join(cmd))
    thing = subprocess.Popen(cmd,
                             # stdout=subprocess.PIPE,
                             )
    time.sleep(10)
    while not engine_started():
        time.sleep(10)
    client = connect_client()
    if start_paused:
        client.simPause(True)
    return thing


def connect_client(client=None):
    if client is None:
        client = airsim.MultirotorClient()  # we are using the multirotor client

    try:
        client.confirmConnection()
    except:
        # failed connection
        return None
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def disconnect_client(client):
    """
    disarms drone and removes api control from client

    """
    client.armDisarm(False)
    client.enableApiControl(False)


def quick_land(client):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 0, 1).join()


def quick_takeoff(client):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 1, 1).join()


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


if __name__ == '__main__':
    if not engine_started():
        thing = start_game_engine(open_gui=True,
                                  start_paused=False,
                                  )
    quit()
    quick = True
    client = connect_client()
    step(client=client,
         seconds=60,
         cmd=lambda: client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, .6, 1),
         pause_after=False,
         )
    quit()
    if quick:
        quick_takeoff(client)
    else:
        client.takeoffAsync().join()
    print('taken off')

    # client.moveByRollPitchYawrateThrottleAsync(0,0,0,1,1).join()

    if quick:
        quick_land(client)
    else:
        client.landAsync()
    print('landed off')
