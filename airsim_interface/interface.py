"""
handles interface with unreal airsim thing
running this file will run airsim
"""

import time

import numpy as np
import airsim, subprocess, math, sys

from airsim_interface.load_settings import get_settings, Keychain, get_fov, get_camera_settings

# do this initially
SETTINGS = get_settings()
CAMERA_SETTINGS = get_camera_settings(sett=SETTINGS)


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
    if project is None:
        project = SETTINGS[Keychain.Defaultproj]

    cmd = [SETTINGS[Keychain.UE4loc], project, '-game', '-windowed']
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


def cconfirmConnection(client, debug=False):
    """
    client.confirmConnection without annoying print statements
    """
    if client.ping():
        if debug: print("Connected!")
    else:
        if debug: print("Ping returned false!")
        return False
    server_ver = client.getServerVersion()
    client_ver = client.getClientVersion()
    server_min_ver = client.getMinRequiredServerVersion()
    client_min_ver = client.getMinRequiredClientVersion()

    ver_info = "Client Ver:" + str(client_ver) + " (Min Req: " + str(client_min_ver) + \
               "), Server Ver:" + str(server_ver) + " (Min Req: " + str(server_min_ver) + ")"

    if server_ver < server_min_ver:
        print(ver_info, file=sys.stderr)
        if debug: print("AirSim server is of older version and not supported by this client. Please upgrade!")
    elif client_ver < client_min_ver:
        print(ver_info, file=sys.stderr)
        if debug: print("AirSim client is of older version and not supported by this server. Please upgrade!")
    else:
        if debug: print(ver_info)
    if debug: print('')
    return True


def connect_client(client=None, vehicle_name=''):
    """
    connects a multirotor client to a particular vehicle, enables api ctrl, and arms it
    """
    if client is None:
        client = airsim.MultirotorClient()  # we are using the multirotor client
    try:
        if not cconfirmConnection(client=client, debug=False):
            return None
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
            if None, runs cmd and pauses after
                in this case, cmd should probably be a .join()
        cmd: ASYNCRHONOUS command to run for frames
            if cmd is not asyncronous, it will run fully then run the sim for seconds (if seconds is not None)
        pause_after: pause after
    Returns:

    """
    # unpause
    if client.simIsPause():
        client.simPause(False)
    cmd()
    if seconds is not None:
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


def get_depth_img(client: airsim.MultirotorClient, camera_name='front', numpee=False):
    """
    obtains depth image
        sometimes there is a bug where width and height are zero, queries until this is not the case
    Args:
        client: client
        camera_name: camera name, if tuple, returns tuple of depth images
        numpee: return numpy array, shaped (height, width)
    Returns:
        output of client.simGetImages
    """
    if type(camera_name) == tuple:
        return tuple(get_depth_img(client=client,
                                   camera_name=t,
                                   numpee=numpee,
                                   )
                     for t in camera_name)
    # airsim.ImageType.OpticalFlow
    img_type = airsim.ImageType.DepthPerspective
    depth_image = client.simGetImages([
        airsim.ImageRequest(camera_name,
                            img_type,
                            True,
                            )
    ])[0]
    while depth_image.height*depth_image.width == 0:
        print('IMG CAPTURE FAILED SOMEHOW? trying to capture image again')
        depth_image = client.simGetImages([
            airsim.ImageRequest(camera_name,
                                img_type,
                                True,
                                )
        ])[0]
    if numpee:
        image_height = depth_image.height
        image_width = depth_image.width
        depth_image = np.array(depth_image.image_data_float, dtype=np.float32).reshape(image_height, image_width)
    return depth_image


def get_of_geo_shape(client: airsim.MultirotorClient, camera_name='front'):
    """
    obtains the shape of the image from of_geo
    Args:
        client: client
        camera_name: guess
    Returns:
        shape tuple, probably (2, 240, 320)
    """
    if type(camera_name) == tuple:
        return tuple(get_of_geo_shape(client=client, camera_name=c) for c in camera_name)
    depth_image = get_depth_img(client=client, camera_name=camera_name)
    return (2, depth_image.height, depth_image.width)

CAMERA_NAME_TO_BASIS={
    #'front':np.identity(3),
    'bottom':np.array([
        [1.,0.,0],
        [0.,0.,-1.],
        [0.,1.,0.],
    ]),
}
def of_geo(client: airsim.MultirotorClient,
           camera_name='front',
           vehicle_name='',
           ignore_angular_velocity=True,
           FOVx=None,
           ):
    """
    PROJECTED optic flow array caluclated from geometric data
        imagines projection (shadow) of every point on a sphere around observer
            each point has a projected relative velocity on this sphere
            consider a FOV rectangle cut out of the sphere, and take the x and y relative velocity of each point
            use these relative velocities to calculate optic flow of each point
    assumes STATIC obstacles, can redo this with dynamic obstacles, but it would be much more annoying
    uses drone's velocity and depth image captured to obtain distance/relative velocity of every point in FOV
        from this, can calculate optic flow
    Args:
        client: client
        camera_name: name of camera, if tuple, returns a tuple of of data
        vehicle_name: guess
        ignore_angular_velocity: whether to ignore angular velocity in calculation
            if True, calculated as if camera is on chicken head
        FOVx: in DEGREES set to a specific value because client.simGetFieldOfView is wrong
            if None, obtains value in /Documents/Airsim/settings.json, or wherever this file is (set in airsim_interface/settings.txt)
    Returns:
        optic flow array, shaped (2,H,W) for better use in CNNs
        x component, y component
    """
    if type(camera_name) == tuple:
        return tuple(of_geo(client=client,
                            camera_name=t,
                            vehicle_name=vehicle_name,
                            ignore_angular_velocity=ignore_angular_velocity,
                            FOVx=FOVx,
                            )
                     for t in camera_name)

    if FOVx is None:
        FOVx = get_fov(camera_settings=CAMERA_SETTINGS,
                       camera_name=camera_name,
                       image_type=airsim.ImageType.DepthPerspective,
                       )

    kinematics = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)

    FOVx = math.radians(FOVx)  # in radians now
    # FOVy = 2*math.atan((image_height/image_width)*math.tan(FOVx/2))

    T = np.array(
        [-kinematics.linear_velocity.y_val, -kinematics.linear_velocity.z_val, kinematics.linear_velocity.x_val])
    if camera_name in CAMERA_NAME_TO_BASIS:
        T=T@CAMERA_NAME_TO_BASIS[camera_name]

    # Rotational velocity (angular velocity)
    omega = np.array(
        [-kinematics.angular_velocity.y_val, -kinematics.angular_velocity.z_val, kinematics.angular_velocity.x_val])

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = get_depth_img(client=client, camera_name=camera_name, numpee=True)
    image_height, image_width = depth_map.shape

    # depth_image = get_depth_img(client=client, camera_name=camera_name)
    # image_width = depth_image.width
    # image_height = depth_image.height
    # Convert depth data to a numpy array and reshape it to the image dimensions
    # depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(image_height, image_width)

    # Assuming these are already defined in your code
    Fx = image_width/(2*math.tan(FOVx/2))  # focal length in pixels (Horizontal = Vertical)
    # Fy = image_height/(2*math.tan(FOVy/2))
    Fy = Fx  # Fx and Fx are the same value

    Z = depth_map  # Depth map

    X_dot, Y_dot, Z_dot = T  # Linear velocities
    p, q, r = omega  # Angular velocities
    if ignore_angular_velocity:
        p, q, r = 0., 0., 0.  # ignore angular motion of drone
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
    geometric_flow = np.stack((Qu, Qv), axis=0)

    # Downsample the geometric flow if needed
    # geometric_flow_ds = geometric_flow[::1, ::1]

    return geometric_flow


if __name__ == '__main__':
    client = connect_client(vehicle_name='')
    # client.moveByRollPitchYawrateThrottleAsync(roll=0, pitch=0, throttle=.6, yaw_rate=0, duration=1, ).join()

    step(client=client,
         seconds=.1,
         cmd=lambda: client.moveByVelocityAsync(vx=1, vy=0, vz=-1,
                                                duration=1, vehicle_name='').join(),
         )

    step(client=client,
         seconds=.1,
         cmd=lambda: client.moveByRollPitchYawrateThrottleAsync(roll=0, pitch=0, throttle=.6, yaw_rate=0,
                                                                duration=.1, ).join(),
         )
    step(client=client,
         seconds=.1,
         cmd=lambda: client.moveByRollPitchYawrateThrottleAsync(roll=0, pitch=0, throttle=.6, yaw_rate=0,
                                                                duration=.1, ).join(),
         )
    print(of_geo(client=client,
                 camera_name='front'))
    print(of_geo(client=client,
                 camera_name='bottom'))
