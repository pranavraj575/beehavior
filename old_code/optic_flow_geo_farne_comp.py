import airsim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math


# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Define initial and target positions
initial_pose = client.simGetVehiclePose().position
target_position = initial_pose + airsim.Vector3r(10, 0, 0)  # Move 10 meters forward

# Set camera parameters for geometric optical flow calculation
focal_length_px = 128.2155  # Example focal length in pixels, set based on your setup

# Data storage for plotting
time_steps = []
farneback_flows = []
geometric_flows = []

# Set up the figure for real-time plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
farneback_ax, geometric_ax = axes
farneback_ax.set_title("Farneback Optical Flow")
farneback_ax.set_xlabel("Pixel X Coordinate")
farneback_ax.set_ylabel("Pixel Y Coordinate")
geometric_ax.set_title("Geometric Optical Flow")
geometric_ax.set_xlabel("Pixel X Coordinate")
farneback_ax.invert_yaxis()
geometric_ax.invert_yaxis()

# Set initial velocity and duration
velocity = 1  # Move forward at 1 m/s
duration = 60  # Move forward for 10 seconds
interval = 0.1 

# Take off to ensure the drone is airborne
client.takeoffAsync().join()

def geo_farn_compar():
     # Capture two consecutive images for optical flow calculation
    prev_image = client.simGetImages([airsim.ImageRequest("bottom", airsim.ImageType.Scene, False, False)])[0]
    prev_img = np.frombuffer(prev_image.image_data_uint8, dtype=np.uint8).reshape(prev_image.height, prev_image.width, 3)
    time.sleep(1 / velocity)  # Wait for the next frame
    curr_image = client.simGetImages([airsim.ImageRequest("bottom", airsim.ImageType.Scene, False, False)])[0]
    curr_img = np.frombuffer(curr_image.image_data_uint8, dtype=np.uint8).reshape(curr_image.height, curr_image.width, 3)
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    ### save image views
    # curr_time=time.time()
    # camera_view_folder_path=r"saved_camera_views"
    # prev_im_path=os.path.join(camera_view_folder_path,f'{curr_time}_previous.png')
    # cur_im_path=os.path.join(camera_view_folder_path,f'{curr_time}_current.png')
    # cv2.imwrite(cur_im_path, curr_img)
    # cv2.imwrite(prev_im_path, prev_img)

    # # Crop to the middle part of the height
    # h, w = prev_gray.shape
    # cropped_height = 30  # Desired height for the middle part
    # middle_start = h // 2 - cropped_height // 2
    # middle_end = h // 2 + cropped_height // 2
    # prev_gray_cropped = prev_gray[middle_start:middle_end, :]
    # curr_gray_cropped = curr_gray[middle_start:middle_end, :]

    # 1. Calculate Farneback optical flow
    farneback_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                  pyr_scale=0.5, levels=3, winsize=9,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Get actual dimensions of the farneback_flow array
    flow_height, flow_width = farneback_flow.shape[:2]

    ### cropped farneback Optic flow 
    # cropped_height = 20  # Desired height for the middle part
    # middle_start = 0 #flow_height // 2 - cropped_height // 2
    # middle_end = 20 #flow_height // 2 + cropped_height // 2
    # cropped_flow = farneback_flow[middle_start:middle_end, :]
    # Pixel selection (subset for plotting)
    pixel_indices = [
        (min(50, flow_width - 1), min(50, flow_height - 1)),
        (min(100, flow_width - 1), min(100, flow_height - 1)),
        (min(150, flow_width - 1), min(150, flow_height - 1)),
        (min(200, flow_width - 1), min(200, flow_height - 1))
    ]

    #print("farneback_flow shape:", farneback_flow.shape)
  
   # Calculate Farneback flow magnitude for the selected pixels
    farneback_flow_magnitudes = [np.sqrt(farneback_flow[y, x, 0]**2 + farneback_flow[y, x, 1]**2) 
                                 for (x, y) in pixel_indices]
    farneback_flows.append(farneback_flow_magnitudes)


    #print(f"Farneback_Flow_height:{flow_height};Farneback_Flow_width:{flow_width}")
    # 2. Calculate Geometric Optical Flow
    # Retrieve real kinematic data from AirSim
    kinematics = client.simGetGroundTruthKinematics()

    # Translational velocity (linear velocity) !!! switch x,y,z velocity due to Airsim setting
    T = np.array([
        -kinematics.linear_velocity.y_val,
        -kinematics.linear_velocity.z_val,  
        kinematics.linear_velocity.x_val
        
    ])

    # Rotational velocity (angular velocity)
    omega = np.array([
        
        -kinematics.angular_velocity.y_val,
        -kinematics.angular_velocity.z_val,
        kinematics.angular_velocity.x_val
        
    ])
    
    height, width = prev_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords - width // 2
    y_coords = y_coords - height // 2
 
    # Retrieve depth map from AirSim (each element is the Z depth for that pixel)
    depth_image = client.simGetImages([airsim.ImageRequest("bottom", airsim.ImageType.DepthPerspective, True)])[0]

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
    
    # Print dimensions to verify
    print(f"Scene Image Shape: {curr_img.shape}")
    print(f"Depth Image Shape: {depth_map.shape}")


    depth_resized = cv2.resize(depth_map, (curr_img.shape[1], curr_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(f"Resized Depth Image Shape: {depth_resized.shape}")

    # # Now use depth_map[y, x] as Z for each (x, y) in the geometric flow calculation
    # u_flow = (-focal_length_px * (T[0] - (x_coords * T[2]) / depth_map) / depth_map +
    #         (y_coords * omega[0] - x_coords * omega[1] - focal_length_px * omega[2]))
    # v_flow = (-focal_length_px * (T[1] - (y_coords * T[2]) / depth_map) / depth_map +
    #         (x_coords * omega[0] - y_coords * omega[1] print(f"Z_height:{Z_height};Z_width:{Z_width}")+ focal_length_px * omega[2]))
 
    ''' test Matt's geometric OF'''
#     X_dot, Y_dot, Z_dot = T#sideways_velocity, vertical_velocity, normal_velocity
#     p, q, r = omega
#     frame_time = 0.153   # Frame duration (10 FPS == 0.1 frame_time)
  
#     # Combine velocities and rotations into a single state vector
#     state_vector = np.array([X_dot, Y_dot, Z_dot, p, q, r])

#     Fx = Fy = focal_length_px
#     Z = depth_resized

#     # Build the matrix
#     Z_height, Z_width = Z.shape
#     #print(f"Z_height:{Z_height};Z_width:{Z_width}")
#     # OF_list = []
#     # Qu_list = []
#     # Qv_list = []
#     geometric_flow = np.zeros((Z_height,Z_width,2))
#     # Define the center of the image
#     u_center = Z_width // 2
#     v_center = Z_height // 2
#     for v in range(Z_height):
#         for u in range(Z_width):
#               # Shift u and v so that the origin is at the center
#             u_shifted = -(u - u_center) #?
#             v_shifted = v_center - v 
#             # print(f"Orignial pixel position x: {u}; y: {v}; After transform x: {u_shifted}; y:{v_shifted}") 
#             z = Z[v, u]
#             a_row1 = [Fx / z, 0, -u_shifted / z, -u_shifted * v_shifted / Fx, -Fx - (u_shifted**2) / Fx, -v_shifted]  
#             a_row2 = [0, Fy / z, -v_shifted / z, -Fy - (v_shifted**2) / Fy, -u_shifted * v_shifted / Fy, u_shifted]
#             # a_row1 = [Fx / z, 0, -u / z, -u * v / Fx, -Fx - (u**2) / Fx, -v]
#             # a_row2 = [0, Fy / z, -v / z, -Fy - (v**2) / Fy, -u * v / Fy, u]
#             Qu = np.matmul(a_row1,state_vector.T)
#             Qv = np.matmul(a_row2,state_vector.T)
#             # print(f"Qu: {Qu},Qv:{Qv}")
#             geometric_flow[v, u, 0] = Qu * frame_time
#             geometric_flow[v, u, 1] = Qv * frame_time
# #             Qu_list.append(Qu)
# #             Qv_list.append(Qv)
# #             OF_list.append([Qu,Qv])  
           

# #    print("Qu_list [last one]:", Qu_list[len(Qu_list)-1])
# #     Get back to work or come back to work :P Hahahahaha
# #     A = np.array([
# #         [Fx / Z, 0, -U / Z, -U * V / Fx, -Fx - (U**2) / Fx, -V],
# #         [0, Fy / Z, -V / Z, -Fy - (V**2) / Fy, -U * V / Fy, U]
# #     ])

     

# #     geometric_flow = np.dstack((Qu_list,Qv_list)) # * focal_length_px * frame_time # np.dstack((Qu_pixels, Qv_pixels))

    '''Another quicker way'''
    # Assuming these are already defined in your code
    Fx = Fy = 128.2155
    Z = depth_map * 0.5 #depth_resized  # Depth map
    #Z = Z.T
    #print("Transposed Z shape:", Z.shape)
    frame_time = 0.1652  # Frame duration
    X_dot, Y_dot, Z_dot = T  # Linear velocities
    p, q, r = omega  # Angular velocities

    # Combine velocities and rotations into a single state vector
    state_vector = np.array([X_dot, Y_dot, Z_dot, p, q, r])

    # Get image dimensions
    Z_height, Z_width = Z.shape
    print(f"depth_image:height:{Z_height};width:{Z_width}")

    # Define the center of the image
    u_center = Z_width // 2
    v_center = Z_height // 2

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(Z_width), np.arange(Z_height))

    # Shift u and v to center the origin
    u_shifted = -(u - u_center)  # Shift horizontally
    v_shifted = v_center - v     # Shift vertically

           
    # Calculate components of the geometric optical flow in a vectorized manner
    a_row1 = np.array([
        Fx / Z,
        np.zeros_like(Z),
        -u_shifted / Z,
        -u_shifted * v_shifted / Fx,
        -Fx - (u_shifted**2) / Fx,
        -v_shifted
    ])

    a_row2 = np.array([
        np.zeros_like(Z),
        Fy / Z,
        -v_shifted / Z,
        -Fy - (v_shifted**2) / Fy,
        -u_shifted * v_shifted / Fy,
        u_shifted
    ])

    # Perform matrix multiplications for all pixels
    Qu = np.tensordot(a_row1.T, state_vector, axes=1).T # Dot product for each pixel
    Qv = np.tensordot(a_row2.T, state_vector, axes=1).T # Dot product for each pixel

    # Multiply by frame_time to scale
    geometric_flow = np.stack((Qu, Qv), axis=-1) * frame_time
    print(f"geometric_flow dimension:{geometric_flow.shape}")

    '''end test'''
    
    

    # Calculate Geometric flow magnitude for the selected pixels
    geometric_flow_magnitudes = [np.sqrt(geometric_flow[y, x, 0]**2 + geometric_flow[y, x, 1]**2) 
                                 for (x, y) in pixel_indices]
    geometric_flows.append(geometric_flow_magnitudes)


    ### cropped geo OF
    #geometric_flow = geometric_flow[middle_start:middle_end, :]
    # Downsample flow fields for clarity
    step = 10  # Step size for downsampling
    farneback_flow_ds = farneback_flow[::step, ::step]
    
    geometric_flow_ds = geometric_flow[::step, ::step]
    x, y = np.meshgrid(np.arange(0, farneback_flow.shape[1], step),
                       
                       np.arange(0, farneback_flow.shape[0], step))
    return x, y, farneback_flow_ds, geometric_flow_ds


def data_logging():

    # Clear previous data on the plot for real-time update
    farneback_ax.clear()
    geometric_ax.clear()
    
    # Update Farneback optical flow plot
    farneback_ax.cla()
    farneback_ax.quiver(x, y, farneback_flow_ds[..., 0], farneback_flow_ds[..., 1],
                        angles='xy', scale_units='xy', scale=0.5, color='r')
    farneback_ax.set_title("Farneback Optical Flow")
    farneback_ax.set_xlabel("Pixel X Coordinate")
    farneback_ax.set_ylabel("Pixel Y Coordinate")
    farneback_ax.invert_yaxis()

    # Update Geometric optical flow plot
    geometric_ax.cla()
    geometric_ax.quiver(x, y, geometric_flow_ds[..., 0], geometric_flow_ds[..., 1],
                        angles='xy', scale_units='xy', scale=0.5, color='b')
    geometric_ax.set_title("Geometric Optical Flow")
    geometric_ax.set_xlabel("Pixel X Coordinate")
    geometric_ax.invert_yaxis()

    # Pause to display the updated plot
    plt.pause(0.01)
   

# # Circular motion parameters
# R = 1.5  # Radius of the circle in meters
# z = -2.0  # Fixed altitude (negative for AirSim's coordinate system)
# speed = 2.0  # Linear speed in m/s
# T = 10.0  # Time for one full circle in seconds
# circle_duration = 3  # Number of circles
# omega = 2 * np.pi / T  # Angular velocity in rad/s
# dt = 1  # Time step for velocity updates


start_time = time.time()
client.simPause(True) 
while time.time() - start_time < duration: # time.time() - start_time < T * circle_duration: #  
    # Command movement for a short interval


 
    # # Calculate current angle based on time
    # elapsed_time = time.time() - start_time
    # theta = omega * elapsed_time

    # # Compute velocity components
    # vx = -R * omega * np.sin(theta)
    # vy = R * omega * np.cos(theta)

    # # Command velocity
    # client.moveByVelocityAsync(vx, vy, 0, dt)

    # # Wait for the next update
    # time.sleep(dt)
    client.simContinueForTime(0.1)

    client.moveByVelocityAsync(velocity, 0, 0, 10)
    client.simPause(True)
    # yaw_rate_deg_per_sec = 2 * (180 / 3.141592653589793)  # Convert 2 rad/s to degrees/s
    # client.rotateByYawRateAsync(yaw_rate_deg_per_sec, 10)
    time.sleep(interval)

    x, y, farneback_flow_ds, geometric_flow_ds = geo_farn_compar() 
    data_logging()
 
    
    # Store time step
    # time_steps.append(t)
client.simPause(False)
# Stop the quadrotor
client.hoverAsync().join()
time.sleep(5)
client.armDisarm(False)
client.enableApiControl(False)

    

#    ## Farneback convert geometric OF
#     # # Parameters (make sure to set these based on your camera setup)
#     #     focal_length_px = 128.2154840042098  # Example focal length in pixels, replace with your value
#     #     fps = 12  # Frames per second, replace with your frame rate                                                                   

#     #     # Extract flow vectors
#     #     flow_x = flow[..., 0]  # Horizontal flow component (u)
#     #     flow_y = flow[..., 1]  # Vertical flow component (v)

#     #     # Convert pixel displacement to angular displacement (radians/frame)
#     #     theta_x = flow_x / focal_length_px
#     #     theta_y = flow_y / focal_length_px

#     #     # Convert angular displacement to angular velocity (radians/second)
#     #     omega_x = theta_x * fps
#     #     omega_y = theta_y * fps

#     #     # Optional: Calculate the total angular velocity magnitude
#     #     angular_velocity_magnitude = np.sqrt(omega_x**2 + omega_y**2)

#     #     # Example to display or use the results
#     #     print("Angular velocity in x (rad/s):", omega_x)
#     #     print("Angular velocity in y (rad/s):", omega_y)
#     #     print("Angular velocity magnitude (rad/s):", angular_velocity_magnitude)

#     #     r_x = 3.7
#     #     r_y = 3.7
#     #     tangential_velocity_x = omega_x * r_x  # Linear equivalent in m/s
#     #     tangential_velocity_y = omega_y * r_y  # Linear equivalent in m/s
