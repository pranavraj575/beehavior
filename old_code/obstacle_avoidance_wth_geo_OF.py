import airsim
import numpy as np
import cv2
import time
import random
from airsim.types import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

#Target position (where quadrotor is heading)
target_x = 18.0  # Target x position
target_y = 0#18.0  # Target y position
target_z = -3.0  # altitude
TARGET_REACH_THRESHOLD = 0.5

# Blending factor to balance target tracking and obstacle avoidance
#BLENDING_FACTOR = 0.4 # tunning the percentage of the obstacle avoidance velocity control parts

# Lists to store trajectory coordinates
x_trajectory = []
y_trajectory = []

# obstacles position
obstacle_positions = [(-3.77, 2.85), (-5.97, 8.55), (-1.77, 7.05), (-1.37, 13.65), (-0.17, 21.35), (6.83, 23.75), (4.03, 16.55), (2.43, 10.05), (1.63, 2.85), (4.03, -1.95), (8.53, -0.15), (5.73, 4.05), (6.33, 8.55), (9.03, 11.45), (9.03, 16.95), (11.43, 23.75), (14.43, 16.95), (11.73, 9.25), (9.03, 4.05), (12.33, 1.65)]
# In airsim the x,y coordinates are opposite with 2-D tra_fig[(2.85, -3.77), (8.55, -5.97), (7.05, -1.77), (13.65,-1.37), (21.35,-0.17),(23.75,6.83),(16.55,4.03),(10.05,2.43),(2.85,1.63),(-1.95,4.03),(-0.15,8.53),(4.05,5.73),(8.55,6.33),(11.45,9.03),(16.95,9.03),(23.75,11.43),(16.95,14.43),(9.25,11.73),(4.05,9.03),(1.65,12.33)] 

# starting point for qua: (-9405.0,-16353.0,10)
# starting point in tuneel: (-11645.0, -22023.0, 10)

#Lists to store velocity (logging data)
V_cmd_x_list = []
V_cmd_y_list = []
V_tgt_x_list = []
V_tgt_y_list = []


def capture_image(client, camera_name):
    """Capture a single image from the specified camera and downsample it."""
    
    response = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
    if response:
        img_data = response[0].image_data_uint8
        if img_data:
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
            image = cv2.resize(image, (320, 240))  # (1080, 720): Resize to 320x240 for performance
            print(f"Captured {camera_name} image, shape: {image.shape}")
            return image
        else:
            print(f"Warning: Empty image data from {camera_name}")
    else:
        print(f"Warning: No response from {camera_name}")
        
    return None

def capture_images_from_all_cameras(client):
    """Capture images sequentially from four cameras: front, left, right, rear."""
    camera_names = ["front", "left", "right", "rear"]
    images = {}

    for camera_name in camera_names:
        image = capture_image(client, camera_name)
        if image is not None:
            images[camera_name] = image
        else:
            print(f"Warning: Could not capture image from {camera_name}.")
    
    return images

def calculate_optic_flow(client, prev_image, curr_image):
    """Calculates the dense optic flow between two images with optimized settings."""
   
    # Convert the images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    prev_gray = prev_gray.astype(np.uint8)
    curr_gray = curr_gray.astype(np.uint8)

    # Crop to the middle part of the height
    h, w = prev_gray.shape
    cropped_height = 72  # Desired height for the middle part
    middle_start = h // 2 - cropped_height // 2
    middle_end = h // 2 + cropped_height // 2
    prev_gray_cropped = prev_gray[middle_start:middle_end, :]
    curr_gray_cropped = curr_gray[middle_start:middle_end, :]

    # Calculate optic flow for the cropped region
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_cropped,
        curr_gray_cropped,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=30,
        iterations=5,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    
    
    # replace farneback with geometric optic flow
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
 
    # Retrieve depth map from AirSim (each element is the Z depth for that pixel)

    depth_image = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True)])[0]
    

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
    depth_map = depth_map[middle_start:middle_end, :]
    # # Print dimensions to verify
    # print(f"Scene Image Shape: {curr_img.shape}")
    # print(f"Depth Image Shape: {depth_map.shape}")


    # Assuming these are already defined in your code
    Fx = Fy = 277 ##184.75 fov 120  # 277 for fov60; fov90: 160.2694 # focal length in pixels (Horizontal = Vertical)
    Z = depth_map  # Depth map
  
    X_dot, Y_dot, Z_dot = T  # Linear velocities
    p, q, r = omega  # Angular velocities

    # Combine velocities and rotations into a single state vector
    state_vector = np.array([X_dot, Y_dot, Z_dot, p, q, r])

    # Get image dimensions
    Z_height, Z_width = Z.shape

    # Define the center of the image
    u_center = Z_width // 2
    v_center = Z_height // 2

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(Z_width), np.arange(Z_height))

    # Shift u and v to center the origin
    u_shifted = -(u - u_center)  # Shift horizontally
    v_shifted = v_center - v     # Shift vertically

           
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
    Qu = np.tensordot(a_row1.T, state_vector, axes=1).T  # Dot product for each pixel
    Qv = np.tensordot(a_row2.T, state_vector, axes=1).T  # Dot product for each pixel

    # print(f"Qu shape:{Qu.shape};Qv shape:{Qv.shape}")

    geometric_flow = np.stack((Qu,Qv), axis= -1) 

    # Downsample the geometric flow if needed
    geometric_flow_ds = geometric_flow[::1, ::1]

    print(f"geometirc opric flow dimension:{geometric_flow_ds.shape}")

    return geometric_flow_ds  #flow
 

def calculate_average_flow(flow):
    """Calculates the average optic flow magnitude."""
    try:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)
        print(f"Average optic flow magnitude: {avg_mag}")
        return avg_mag
    except Exception as e:
        print(f"Error calculating average flow: {e}")
        return None




# Function to calculate looming using left-right optical flow comparison
def loom_left_right(flow):
    """Calculate the looming value by comparing left and right halves of the flow field."""
    print("Inside loom_left_right function")
    h, w = flow.shape[:2]
    half_width = int(w / 2)

    left_half = flow[:, :half_width, :]
    right_half = flow[:, half_width:, :]

    print(f"Flow shape: {flow.shape}, Left half shape: {left_half.shape}, Right half shape: {right_half.shape}")

    if left_half.size == 0 or right_half.size == 0:
        print("Error: Left or right half of the flow is empty!")
        return 0

    # Sum the magnitudes of the flow vectors for each half
    left_sum = np.sum(np.linalg.norm(left_half, axis=2))
    right_sum = np.sum(np.linalg.norm(right_half, axis=2))

    # Normalize by the number of pixels to get average flow per pixel
    num_left_pixels = left_half.shape[0] * left_half.shape[1]
    num_right_pixels = right_half.shape[0] * right_half.shape[1]

    left_avg = left_sum / num_left_pixels
    right_avg = right_sum / num_right_pixels

    print(f"Left Half Avg Flow: {left_avg}, Right Half Avg Flow: {right_avg}")

    # Calculate the looming value as the difference in average flow between the two halves
    loom_value = left_avg - right_avg
    print(f"Looming Value: {loom_value}")
    return loom_value



def calculate_target_velocity(current_position, target_position, MAX_SPEED=1.0):

    delta_x = target_position[0] - current_position[0]
    delta_y = target_position[1] - current_position[1]

    print(f"Current position: ({current_position[0]},{current_position[1]})")

    # Calculate the distance to the target
    distance_to_target = np.sqrt(delta_x**2 + delta_y**2)

    # Normalize the direction vector (delta_x, delta_y) to get unit direction
    if distance_to_target > TARGET_REACH_THRESHOLD:
        unit_x = delta_x / distance_to_target
        unit_y = delta_y / distance_to_target
    else:
        print("Target is within reach threshold; setting velocities to zero.")
        unit_x, unit_y = 0, 0  # No movement if already at the target

    # Compute target velocities, ##capping to max_speed unit_x/y * min(distance_to_target,max_speed)
    V_tgt_x = unit_x * distance_to_target
    V_tgt_y = unit_y * distance_to_target

    norm = np.sqrt(V_tgt_x**2 + V_tgt_y**2)
    if norm > 0:
        V_tgt_x /= norm
        V_tgt_y /= norm

    V_tgt_x *= MAX_SPEED
    V_tgt_y *= MAX_SPEED

    print(f"target speed: {V_tgt_x,V_tgt_y}")

    return V_tgt_x, V_tgt_y



def calculate_velocity_commands(V_tgt_x, V_tgt_y, L_A, L_F, L_L, L_R, Q_A, Q_F, Q_L, Q_R, V_x, V_y, k0=4, k1=5,MAX_SPEED=1):
    """Calculate the velocity commands based on loom and optic flow signals."""
    V_x_safe = max(abs(V_x), 0.2)
    V_y_safe = max(abs(V_y), 0.2)

    adjustment_x = k0 * (L_A - L_F) / V_x_safe + k1 * (Q_A - Q_F) / V_x_safe
    adjustment_y = k0 * (L_L - L_R) / V_y_safe + k1 * (Q_L - Q_R) / V_y_safe

    print(f"detail adjustments part: loom difference front and back{L_A - L_F}")

    norm = np.sqrt(adjustment_x**2 + adjustment_y**2)
    if norm > 0:
        adjustment_x /= norm
        adjustment_y /= norm

    adjustment_x *= MAX_SPEED
    adjustment_y *= MAX_SPEED


    # adjustment_x = np.clip(adjustment_x, -5.0, 5.0)
    # adjustment_y = np.clip(adjustment_y, -5.0, 5.0)

    # V_cmd_x = (1-BLENDING_FACTOR) * V_tgt_x + BLENDING_FACTOR * adjustment_x
    # V_cmd_y = (1-BLENDING_FACTOR) * V_tgt_y + BLENDING_FACTOR * adjustment_y


    V_cmd_x = V_tgt_x + adjustment_x
    V_cmd_y = V_tgt_y + adjustment_y

    print(f"adjustment parts of the speeds: {adjustment_x,adjustment_y}")


    # norm = np.sqrt(V_cmd_x**2 + V_cmd_y**2)
    # if norm > 0:
    #     V_cmd_x /= norm
    #     V_cmd_y /= norm

    # V_cmd_x *= MAX_SPEED
    # V_cmd_y *= MAX_SPEED

    # if V_cmd_x > 1 or V_cmd_x < -1:
    #     V_cmd_x = V_cmd_x/abs(V_cmd_x)
        
    # if V_cmd_y > 1 or V_cmd_y < -1:
    #     V_cmd_y = V_cmd_y/abs(V_cmd_y)

    # V_cmd_x = np.clip(V_cmd_x, -1.0, 1.0)
    # V_cmd_y = np.clip(V_cmd_y, -1.0, 1.0)
       
        
             
    return V_cmd_x, V_cmd_y



def adjust_drone_path(client, loom_data, flow_data, V_tgt_x=0, V_tgt_y=0):
    """Adjusts the drone's path based on loom and optic flow data to avoid obstacles."""
    # Ensure we have data for all four cameras
    if len(loom_data) == 4 and len(flow_data) == 4:
        L_F, L_L, L_R, L_A = loom_data
        Q_F, Q_L, Q_R, Q_A = flow_data
        
        # Get current state
        state = client.getMultirotorState()

        current_position = (
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        )
        
        # Record the x and y coordinates for trajectory
        x_trajectory.append(current_position[0])
        y_trajectory.append(current_position[1])

        V_tgt_x, V_tgt_y = calculate_target_velocity(current_position, (target_x, target_y, target_z))

        # Current velocities (replace with actual drone velocities)
        V_x = state.kinematics_estimated.linear_velocity.x_val
        V_y = state.kinematics_estimated.linear_velocity.y_val
        print(f"Previous speed:V_x:{V_x},V_y:{V_y}")
        
        # Calculate velocity commands
        V_cmd_x, V_cmd_y = calculate_velocity_commands(V_tgt_x, V_tgt_y, 
                                                       L_A, L_F, L_L, L_R, 
                                                       Q_A, Q_F, Q_L, Q_R, 
                                                       V_x, V_y)

        # Get the current altitude of the drone (Z-axis is positive for ascending)
        
        current_altitude = current_position[2]  #state.kinematics_estimated.position.z_val
       
       
        print(f"Current speed: V_x:{V_cmd_x},V_y:{V_cmd_y}")
        # Move the drone using the calculated velocity commands
        # yaw_angle = (np.arctan2(V_cmd_y,V_cmd_x))
        # yaw_angle = np.float64(np.degrees(yaw_angle))

        # client.moveByVelocityAsync(V_cmd_x,V_cmd_y,0,duration=1, drivetrain= DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(is_rate=False, yaw_or_rate=yaw_angle))
        
        '''logging velocity changing'''
        V_cmd_x_list.append(V_cmd_x)
        V_cmd_y_list.append(V_cmd_y)
        V_tgt_x_list.append(V_tgt_x)
        V_tgt_y_list.append(V_tgt_y)

        ### moving commands:
        client.moveByVelocityAsync(V_cmd_x,V_cmd_y,0,duration=1, drivetrain= DrivetrainType.ForwardOnly)
        

        time.sleep(0.9)
        #print(f"current altitude:{current_altitude}")
        # if current_altitude > -2.9 or current_altitude < -3.1:
        #    client.moveToZAsync(-3, 1.5)
        if current_altitude > target_z + 0.1 or current_altitude < target_z - 0.1:
            client.moveToZAsync(target_z, 1.5)
        
        # distance_to_target = np.sqrt((current_position[0] - target_x)**2 + (current_position[1] - target_y)**2)
        # if distance_to_target <= TARGET_REACH_THRESHOLD:
        #     print(f"Target reached at position: x={current_position[0]}, y={current_position[1]}")


        # return V_cmd_x, V_cmd_y, V_tgt_x, V_tgt_y

def process_camera_flow(client, camera_name, prev_image, curr_image):
    """Calculate optic flow for a single camera."""

    try:
        flow = calculate_optic_flow(client, prev_image, curr_image)
        if flow is not None:
            avg_flow = calculate_average_flow(flow)
            loom = loom_left_right(flow)
            return flow, loom, avg_flow
        else:
            return None, None    
        
    except Exception as e:
        print(f"Error in camera {camera_name} flow calculation: {e}")
        return None, None
    

# def plot_trajectory(x_trajectory, y_trajectory,obstacle_positions, obstacle_size=0.5): #
#     """Plot the recorded trajectory in 2D (x-y coordinates)."""   
#     plt.clf()
#     plt.plot( y_trajectory, x_trajectory, color='b', linestyle='-', label='Trajectory')  #mmarker='o'
#     #  # Plot obstacles as red 'X'
#     # for obstacle in obstacle_positions:
#     #     plt.scatter(obstacle[0], obstacle[1], color='r', marker='x', s=100, label='Obstacle')
    
#     for (ox, oy) in obstacle_positions:
#     # Create a square centered at (ox, oy)
#         obstacle = patches.Rectangle((ox - obstacle_size/2, oy - obstacle_size/2), 
#                                     obstacle_size, obstacle_size, linewidth=1, 
#                                     edgecolor='r', facecolor='none') #, label='Obstacle'
#         plt.gca().add_patch(obstacle)



#     # # Wall positions
#     # wall_1_x = -3.73
#     # wall_2_x = 3.97
#     # y_min, y_max = -10, 50  # Range of y values for walls
#     # wall_1 = patches.Rectangle((wall_1_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)
#     # wall_2 = patches.Rectangle((wall_2_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)

#     # # Add the walls to the plot
#     # plt.gca().add_patch(wall_1)
#     # plt.gca().add_patch(wall_2)

#     plt.ylim(0,20)
#     plt.xlim(0,20)
#     plt.xlabel('Y Coordinate')
#     plt.ylabel('X Coordinate')
#     plt.title('Quadrotor 2D Trajectory (X-Y)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
   



def plot_trajectory(trajectory_fig, trajectory_ax, x_trajectory, y_trajectory, obstacle_positions, obstacle_size=0.5):
    """Continuously update the trajectory plot."""
    trajectory_ax.clear()  # Clear previous data on the trajectory plot

    # Plot trajectory
    trajectory_ax.plot(y_trajectory, x_trajectory, color='b', linestyle='-', label='Trajectory')

    # # Plot obstacles
    # for (ox, oy) in obstacle_positions:
    #     obstacle = patches.Rectangle(
    #         (ox - obstacle_size / 2, oy - obstacle_size / 2),
    #         obstacle_size, obstacle_size, linewidth=1, 
    #         edgecolor='r', facecolor='none'
    #     )
    #     trajectory_ax.add_patch(obstacle)

     
    # Wall positions
    wall_1_x = -2.5
    wall_2_x = 2.5
    y_min, y_max = -10, 50  # Range of y values for walls
    wall_1 = patches.Rectangle((wall_1_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)
    wall_2 = patches.Rectangle((wall_2_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)

    # Add the walls to the plot
    plt.gca().add_patch(wall_1)
    plt.gca().add_patch(wall_2)

    # Set axis limits and labels
    trajectory_ax.set_xlim(-5, 10)
    trajectory_ax.set_ylim(-5, 20)
    trajectory_ax.set_xlabel('Y Coordinate')
    trajectory_ax.set_ylabel('X Coordinate')
    trajectory_ax.set_title('Quadrotor 2D Trajectory (X-Y)')
    trajectory_ax.legend()
    trajectory_ax.grid(True)

    # Pause to update plot
    trajectory_fig.canvas.draw()
    trajectory_fig.canvas.flush_events()


def data_logging(time_steps, ax1, ax2):

    # Ensure all lists have the same length
    if len(time_steps) != len(V_cmd_x_list) or len(time_steps) != len(V_tgt_x_list):
        print("Data lengths are inconsistent. Skip    plt.show(ping plot update.")
        return

    # Clear previous data on the plot for real-time update
    ax1.clear()
    ax2.clear()
    
    # Plot V_cmd and V_tgt velocities in x and y directions
    ax1.plot(time_steps, V_cmd_x_list, label="V_cmd_x", linestyle='-', marker='o')
    ax1.plot(time_steps, V_tgt_x_list, label="V_tgt_x", linestyle='--', marker='x')
    ax1.set_ylabel("Velocity X")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(time_steps, V_cmd_y_list, label="V_cmd_y", linestyle='-', marker='o')
    ax2.plot(time_steps, V_tgt_y_list, label="V_tgt_y", linestyle='--', marker='x')
    ax2.set_ylabel("Velocity Y")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # Refresh the plot
    plt.pause(0.01)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.moveToZAsync(-3,1.5).join()

    time_steps = []
    start_time = time.time()  # Record start time
    frame_count = 0

    prev_images = capture_images_from_all_cameras(client)
    if len(prev_images) < 4:
        print("Error: Not all cameras captured successfully in the first frame.")
        return

    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))    
    # Initialize trajectory and velocity figures
    trajectory_fig, trajectory_ax = plt.subplots()
    # client.simPause(True)

    while True:  # Limit the loop to 100 frames to avoid performance issues
        
        curr_images = capture_images_from_all_cameras(client)
        time_flag1 = time.time()
        sim_time_star_frm = client.getMultirotorState().timestamp       
        loom_data = []
        flow_data = []
        
        curr_position = client.simGetVehiclePose().position  # Get current position
        distance_to_target = math.sqrt((curr_position.x_val - target_x)**2 + 
                                       (curr_position.y_val - target_y)**2)
         # Check if the target is reached based on distance
        if distance_to_target < TARGET_REACH_THRESHOLD:
            print("Target reached. Exiting loop.")
            break  # Exit the loop once the target is reached

        # Calculate optic flow for every frame
        print(f"Processing frame {frame_count} for optic flow calculations")
        for camera_name in ["front", "left", "right", "rear"]:
            flow, loom, avg_flow = process_camera_flow(client, camera_name, prev_images[camera_name], curr_images[camera_name])
            if loom is not None and avg_flow is not None:
                loom_data.append(loom)
                flow_data.append(avg_flow)
            else:
                print(f"Error: Incomplete data for {camera_name} at frame {frame_count}")
        
          
        # Only adjust path if we have valid data from all cameras
        if len(loom_data) == 4 and len(flow_data) == 4:
            # client.simContinueForTime(10)
            adjust_drone_path(client, loom_data, flow_data)

            current_time = time.time() - start_time  # Calculate elapsed time in seconds
            time_steps.append(current_time)

            # Log and plot the velocities in real-time
            data_logging(time_steps, ax1, ax2)
            # client.simPause(True)
     
        else:
            print(f"Warning: Loom or Flow data incomplete, skipping frame {frame_count}.")

        prev_images = curr_images  # Update previous images for the next iteration
        # prev_time = curr_time  # Update time
        frame_count += 1
        print(f"FRAME: {frame_count}")

        plot_trajectory(trajectory_fig, trajectory_ax, x_trajectory, y_trajectory,obstacle_positions, obstacle_size=1)
        

        time_flag2 = time.time()
        sim_time_end_frm = client.getMultirotorState().timestamp
        real_time_dur = time_flag2 - time_flag1
        print(f"Real time elapsed:{real_time_dur} seconds")
        fps = 1/((sim_time_end_frm - sim_time_star_frm) / 1e9)
        print(f"Simulation frame rate:{fps} FPS")
    
    # client.simPause(False) 
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # plot_trajectory(x_trajectory, y_trajectory)
    
    # Outside the main loop, at the end of the program:
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plo


if __name__ == "__main__":
    main()
    