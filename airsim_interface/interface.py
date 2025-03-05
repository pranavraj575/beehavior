import airsim, subprocess
from airsim_interface.load_settings import get_settings, Keychain


def engine_started():
    return connect_client() is not None


def start_engine():
    sett = get_settings()
    thing = subprocess.Popen([sett[Keychain.UE4loc]], stdout=subprocess.PIPE)
    return thing


def connect_client():
    client = airsim.MultirotorClient()  # we are using the multirotor client
    try:
        client.confirmConnection()
    except:
        # failed connection
        return None
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def quick_land(client):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 0, 1).join()


def quick_takeoff(client):
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 1, 1).join()


if __name__ == '__main__':
    if not engine_started():
        thing = start_engine()
    quick = True
    client = connect_client()
    if quick:
        quick_takeoff(client)
    else:
        client.takeoffAsync().join()
    print('taken off')

    client.moveByRollPitchYawrateThrottleAsync(0,3.14,0,1,1).join()

    if quick:
        quick_land(client)
    else:
        client.landAsync()
    print('landed off')
