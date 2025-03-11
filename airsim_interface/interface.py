"""
handles interface with unreal airsim thing
running this file will run airsim
"""

import time

import numpy as np
import airsim, subprocess, math

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
        subprocess object
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
    """
    connects a multirotor client to a particular vehicle, enables api ctrl, and arms it
    """
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


def step(client, seconds=.5, cmd=lambda: None, pause_after=True):
    """
    steps a simulation
    Args:
        seconds: number of seconds to continue for
        cmd: ASYNCRHONOUS command to run for frames
            if cmd is not asyncronous, it will run fully on a paused simulation
    Returns:

    """
    # unpause
    if client.simIsPause():
        client.simPause(False)
    cmd()
    client.simContinueForTime(seconds=seconds)
    # repause
    if pause_after:
        client.simPause(True)


def move_along_pth(client: airsim.MultirotorClient, pth, v=1., vehicle_name=''):
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


def get_of_geo_shape(client: airsim.MultirotorClient, camera_name='front'):
    """
    obtains the shape of the image from of_geo
    Args:
        client: client
        camera_name: guess
    Returns:
        shape tuple, probably (240, 360, 2)
    """
    depth_image = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True)])[0]
    return (depth_image.height, depth_image.width, 2)


def of_geo(client: airsim.MultirotorClient, camera_name='front', vehicle_name='', FOVx=60):
    """
    optic flow array caluclated from geometric data
    assumes STATIC obstacles, can redo this with dynamic obstacles, but it would be much more annoying
    uses drone's velocity and depth image captured to obtain distance/relative velocity of every point in FOV
        from this, can calculate optic flow
    Args:
        client: client
        camera_name: name of camera
        vehicle_name: guess
        FOVx: in DEGREES set to a specific value because client.simGetFieldOfView is wrong
    Returns:
        optic flow array, shaped (H,W,2)
    """
    depth_image = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True)])[0]
    kinematics = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    image_width = depth_image.width
    image_height = depth_image.height

    FOVx = math.radians(FOVx)  # in radians now
    # FOVy = 2*math.atan((image_height/image_width)*math.tan(FOVx/2))

    T = np.array(
        [-kinematics.linear_velocity.y_val, -kinematics.linear_velocity.z_val, kinematics.linear_velocity.x_val])

    # Rotational velocity (angular velocity)
    omega = np.array(
        [-kinematics.angular_velocity.y_val, -kinematics.angular_velocity.z_val, kinematics.angular_velocity.x_val])

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(image_height, image_width)

    # Assuming these are already defined in your code
    Fx = image_width/(2*math.tan(FOVx/2))  # focal length in pixels (Horizontal = Vertical)
    # Fy = image_height/(2*math.tan(FOVy/2))
    Fy = Fx  # Fx and Fx are the same value

    Z = depth_map  # Depth map

    X_dot, Y_dot, Z_dot = T  # Linear velocities
    p, q, r = omega  # Angular velocities

    # Combine velocities and rotations into a single state vector
    state_vector = np.array([X_dot, Y_dot, Z_dot, p, q, r])

    # Get image dimensions
    Z_height, Z_width = Z.shape

    # Define the center of the image
    u_center = Z_width//2
    v_center = Z_height//2

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(Z_width), np.arange(Z_height))

    # Shift u and v to center the origin
    u_shifted = -(u - u_center)  # Shift horizontally
    v_shifted = v_center - v  # Shift vertically

    a_row1 = np.array([
        Fx/Z,
        np.zeros_like(Z),
        -u_shifted/Z,
        -u_shifted*v_shifted/Fx,
        -Fx - (u_shifted**2)/Fx,
        -v_shifted,
    ])

    a_row2 = np.array([
        np.zeros_like(Z),
        Fy/Z,
        -v_shifted/Z,
        -Fy - (v_shifted**2)/Fy,
        -u_shifted*v_shifted/Fy,
        u_shifted,
    ])
    # Perform matrix multiplications for all pixels
    Qu = np.tensordot(a_row1.T, state_vector, axes=1).T  # Dot product for each pixel
    Qv = np.tensordot(a_row2.T, state_vector, axes=1).T  # Dot product for each pixel

    # print(f"Qu shape:{Qu.shape};Qv shape:{Qv.shape}")
    # Multiply by frame_time to scale
    geometric_flow = np.stack((Qu, Qv), axis=-1)

    # Downsample the geometric flow if needed
    # geometric_flow_ds = geometric_flow[::1, ::1]

    return geometric_flow


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
