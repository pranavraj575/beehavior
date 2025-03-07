import math
import airsim
import numpy as np


def of_geo(client, image_width=320, image_height=240, fov=60):
    # TODO: read camera settings from client
    depth_image = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True)])[0]
    kinematics = client.simGetGroundTruthKinematics()

    T = np.array(
        [-kinematics.linear_velocity.y_val, -kinematics.linear_velocity.z_val, kinematics.linear_velocity.x_val])

    # Rotational velocity (angular velocity)
    omega = np.array(
        [-kinematics.angular_velocity.y_val, -kinematics.angular_velocity.z_val, kinematics.angular_velocity.x_val])

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)

    focal_length_x = image_width/(2*math.tan(fov/2))

    vertical_FOV = 2*math.atan((math.tan(fov/2))/(image_width/image_height))
    focal_length_y = image_height/(2*math.tan(vertical_FOV/2))
    # Assuming these are already defined in your code
    Fx = focal_length_x  # focal length in pixels (Horizontal = Vertical)
    Fy = focal_length_y
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
        -v_shifted
    ])

    a_row2 = np.array([
        np.zeros_like(Z),
        Fy/Z,
        -v_shifted/Z,
        -Fy - (v_shifted**2)/Fy,
        -u_shifted*v_shifted/Fy,
        u_shifted
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
