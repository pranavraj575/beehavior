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
target_x = 20.0  # Target x position
target_y = 0.0  # Target y position
target_z = -3.0  # altitude
TARGET_REACH_THRESHOLD = 1

# Blending factor to balance target tracking and obstacle avoidance
#BLENDING_FACTOR = 0.4 # tunning the percentage of the obstacle avoidance velocity control parts

# Lists to store trajectory coordinates
x_trajectory = []
y_trajectory = []

# obstacles position
obstacle_positions = [(-3.77, 2.85), (-5.97, 8.55), (-1.77, 7.05), (-1.37, 13.65), (-0.17, 21.35), (6.83, 23.75), (4.03, 16.55), (2.43, 10.05), (1.63, 2.85), (4.03, -1.95), (8.53, -0.15), (5.73, 4.05), (6.33, 8.55), (9.03, 11.45), (9.03, 16.95), (11.43, 23.75), (14.43, 16.95), (11.73, 9.25), (9.03, 4.05), (12.33, 1.65)]
tunnel_obstacles = [(-1.04,1.75),(0.66,12.65)]
# In airsim the x,y coordinates are opposite with 2-D tra_fig[(2.85, -3.77), (8.55, -5.97), (7.05, -1.77), (13.65,-1.37), (21.35,-0.17),(23.75,6.83),(16.55,4.03),(10.05,2.43),(2.85,1.63),(-1.95,4.03),(-0.15,8.53),(4.05,5.73),(8.55,6.33),(11.45,9.03),(16.95,9.03),(23.75,11.43),(16.95,14.43),(9.25,11.73),(4.05,9.03),(1.65,12.33)] 

# starting point for qua: (-9405.0,--16353.0,10)
# starting point in tunel: (-9405.0, -22146.0, 10)

V_cmd_x_list = []
V_cmd_y_list = []
V_tgt_x_list = []
V_tgt_y_list = []

def capture_image(client, camera_name):
    """Capture a single image from the specified camera and downsample it."""
    
    depth_image = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True)])[0]
        

    if depth_image:

        # Convert depth data to a numpy array and reshape it to the image dimensions
        depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
        
        return depth_map

    else:
        print(f"Warning: No response from {camera_name}")
        
    return None

## Uses six depth_map segments from three cameras (right, front, left) each split into two segments.
def capture_images_from_all_cameras(client):
    camera_names = ["right", "front", "left"]
    depth_maps = {}

    for camera_name in camera_names:
        depth_map = capture_image(client, camera_name)
        if depth_map is not None:
            depth_maps[camera_name] = depth_map
        else:
            print(f"Warning: Could not capture image from {camera_name}.")
    
    return depth_maps

def segment_image(images):
   
    segments = []

    for img in images.values():
        h, w = img.shape
        mid_w = w // 2  # Midpoint for horizontal division

        # Divide the image into two horizontal segments
        segment_1 = img[:, :mid_w]
        segment_2 = img[:, mid_w:]

        # Add the segments to the list
        segments.extend([segment_1, segment_2])
        
    return segments

def process_segments(client, curr_segments, V_cmd_x, V_cmd_y):
  
    loom_data = []
    flow_data = []

    # Crop to the middle part of the height
    cropped_height = 10  # Desired height for the middle part
    h, w = curr_segments[0].shape
    middle_start = h // 2 - cropped_height // 2
    middle_end = h // 2 + cropped_height // 2
    
  
    for i in range(len(curr_segments)):
        curr_segment = curr_segments[i]

        depth_map = curr_segment[middle_start:middle_end, :]
        # # Print dimensions to verify
        # print(f"Scene Image Shape: {curr_img.shape}")
        # print(f"Depth Image Shape: {depth_map.shape}")

         # replace farneback with geometric optic flow
        kinematics = client.simGetGroundTruthKinematics()
        angle = np.deg2rad(30)

        # coordinates transformation (drone --> right image plane)
        if i in (0,1):
            T = np.array([
                (-kinematics.linear_velocity.x_val * np.cos(angle) + kinematics.linear_velocity.y_val * np.sin(angle)),
                -kinematics.linear_velocity.z_val,  
                (kinematics.linear_velocity.x_val * np.sin(angle) + kinematics.linear_velocity.y_val * np.cos(angle))
                
            ])

            omega = np.array([
                
                (-kinematics.angular_velocity.x_val * np.cos(angle) + kinematics.angular_velocity.y_val * np.sin(angle)),
                -kinematics.angular_velocity.z_val,  
                (kinematics.angular_velocity.x_val * np.sin(angle) + kinematics.angular_velocity.y_val * np.cos(angle))
                
            ])

        # coordinates transformation (drone --> front image plane)
        elif i in (2,3):
        # Translational velocity (linear velocity) !!! Transforming drone's coordinates to image plane coordinates
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

        # coordinates transformation (drone --> left image plane)
        elif i in (4,5):

            T = np.array([
                (kinematics.linear_velocity.x_val * np.cos(angle) + kinematics.linear_velocity.y_val * np.sin(angle)),
                -kinematics.linear_velocity.z_val,  
                (kinematics.linear_velocity.x_val * np.sin(angle) - kinematics.linear_velocity.y_val * np.cos(angle))
                
            ])

            omega = np.array([
                
                (kinematics.angular_velocity.x_val * np.cos(angle) + kinematics.angular_velocity.y_val * np.sin(angle)),
                -kinematics.angular_velocity.z_val,  
                (kinematics.angular_velocity.x_val * np.sin(angle) - kinematics.angular_velocity.y_val * np.cos(angle))
                
            ])

        else:
            print("image plane coordinates transformation FAILED! Segments index OUT OF RANGE!")

        flow = geometric_optic_flow(depth_map, T, omega)

        # print(f"geometirc opric flow dimension:{flow.shape}")
        # Calculate average flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_flow = np.mean(mag)  # Average magnitude as scalar

        flow_data.append(avg_flow)  # Store as scalar
    # Calculate looming using flow magnitudes
    loom_data = calculate_true_loom(client, curr_segments, max_loom_threshold=100) 

    # Debugging output to confirm structure
    print(f"Loom data: {loom_data}")
    print(f"Flow data: {flow_data}")

    return loom_data, flow_data

def geometric_optic_flow(depth_map, T, omega):
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

        return geometric_flow_ds

def calculate_true_loom(client, depth_map_segments, max_loom_threshold=100):
    """
    Calculate true loom (only moving direction) as the normal velocity (perpendicular to obstacles) divided by depth.
    """

    state = client.getMultirotorState()
    V_x = state.kinematics_estimated.linear_velocity.x_val
    V_y = state.kinematics_estimated.linear_velocity.y_val

    if depth_map_segments is None or len(depth_map_segments) != 6:
        raise ValueError("Six depth map segments are required to compute true loom.")
    
    if V_x is None or V_y is None:
        print("No commanded velocity provided, using front view loom.")
        front_depth_map = [depth_map_segments[2], depth_map_segments[3]]  # Using the two front segments
        valid_front_depth = np.concatenate([dm[dm > 0] for dm in front_depth_map])
        if valid_front_depth.size == 0:
            raise ValueError("No valid depth data in front segments.")
        front_loom = V_x / np.mean(valid_front_depth)  # Compute front loom
        front_loom_clipped = np.clip(front_loom, -max_loom_threshold, max_loom_threshold)
        print(f"Front view loom (initial frame): {front_loom_clipped}")
        return front_loom_clipped
    
    # Section centers (in radians): [pi/12, pi/4, ...]
    section_centers = np.linspace(np.pi / 12, 11 * np.pi / 12, 6)
    section_width = np.pi / 6  # 30 degrees per section

    # Compute moving direction
    moving_direction = np.arctan2(V_x, V_y) % (2 * np.pi)

    # Find nearest section
    section_diffs = np.abs(section_centers - moving_direction)
    nearest_section = np.argmin(section_diffs)
    
    # Compute normal velocity (perpendicular to moving direction)
    normal_velocity = np.sqrt(V_x**2 + V_y**2) 
    
    # Compute loom only for the nearest section
    loom_values = [0] * 6  # Initialize all looms as zero
    valid_depth_mask = depth_map_segments[nearest_section] > 0
    if np.any(valid_depth_mask):
        segment_loom = normal_velocity / np.mean(depth_map_segments[nearest_section][valid_depth_mask])
        loom_values[nearest_section] = segment_loom
    
    # Compute overall loom
    loom_value = loom_values[nearest_section]
    
    # Clip loom to avoid extreme values
    #loom_values = np.clip(loom_values, -max_loom_threshold, max_loom_threshold)
    
    print(f"Loom Value (Section {nearest_section}): {loom_value}")
    return loom_values



# def visualize_optic_flow(image, flow):
#     """
#     Visualize optic flow by drawing arrows.
#     """
#     print(f"Inside visualizing optic flow section")
#     h, w = flow.shape[:2]
#     step = 8  # Arrow step size to reduce clutter

#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fx, fy = flow[y, x]
#             start_x, start_y = x, y + 115
#             end_x, end_y = int(start_x + fx), int(start_y + fy)
#             #end_x, end_y = int(x + fx), int(y + fy)
#             cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1, tipLength=0.5)

#     return image

# From the current and target position calculating the target velocity vector
def calculate_target_velocity(current_position, target_position, MAX_SPEED=2.0):
   
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

    V_tgt_x = np.clip(V_tgt_x, -MAX_SPEED, MAX_SPEED)
    V_tgt_y = np.clip(V_tgt_y, -MAX_SPEED, MAX_SPEED)

    # print(f"target speed: {V_tgt_x,V_tgt_y}")

    return V_tgt_x, V_tgt_y

def velocity_controller(loom_data, flow_data, V_x, V_y, flow_weight = 0.5, loom_weight = 500, SPEED_CAP = 0.5):
   
    # Directions: [right, front, left] / 2 for each
    num_directions = len(loom_data)
    direction_weights = [-15, -5, -1, -1, -5, -10] ##[-5] * num_directions #* num_directions  # Adjust weights for directional priority: right to left

    # # Normalize flow and loom data
    # norm_flow = [flow / max(flow_data) if max(flow_data) > 0 else 0 for flow in flow_data]
    # norm_loom = [loom / max(loom_data) if max(loom_data) > 0 else 0 for loom in loom_data]

    # Compute adjustments based on normalized flow and loom
    adjustments = [direction_weights[i] * (flow_weight * flow_data[i] + loom_weight * loom_data[i]) for i in range(num_directions)]

    # Determine velocity components
    angle_step = 180 / num_directions
    V_x_adjust, V_y_adjust = 0, 0

    # Calculate the center angles for each segment
    start_angle = 15  # Degrees for the center of the front-left segment
    direction_angles = [np.deg2rad(start_angle + i * angle_step) for i in range(6)]
    # direction_angles = [np.pi*(11/12), np.pi*(9/12), np.pi*(7/12), np.pi *(5/12), np.pi*(1/4), np.pi*(1/12)]

    # Calculate velocity adjustments using center angles
    for i, adjustment in enumerate(adjustments):
        angle = direction_angles[i]  # Center angle for this segment
        V_y_adjust += adjustment * np.cos(angle)  # Horizontal component
        V_x_adjust += adjustment * np.sin(angle)  # Vertical component

    V_x_adjust = np.clip(V_x_adjust, -SPEED_CAP, SPEED_CAP)
    V_y_adjust = np.clip(V_y_adjust, -SPEED_CAP, SPEED_CAP) 

    # Combine adjustments with existing velocity
    V_cmd_x = V_x + V_x_adjust
    V_cmd_y = V_y + V_y_adjust

    return V_cmd_x, V_cmd_y

# def saccades(client, V_cmd_x, V_cmd_y):
#     if V  < thershold:  # saccades mode (maintain fixed location hovering and looking around to gain information)
        

def adjust_drone_path(client, loom_data, flow_data, V_tgt_x=0, V_tgt_y=0):
    """Adjusts the drone's path based on loom and optic flow data to avoid obstacles."""
    # Ensure we have data for all four cameras
    if len(loom_data) == 6 and len(flow_data) == 6:
              
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
        # V_x = state.kinematics_estimated.linear_velocity.x_val
        # V_y = state.kinematics_estimated.linear_velocity.y_val
        # print(f"Previous speed:V_x:{V_x},V_y:{V_y}")
        
        # Calculate velocity commands
        V_cmd_x, V_cmd_y = velocity_controller(loom_data,flow_data,V_tgt_x,V_tgt_y)

        # Get the current altitude of the drone (Z-axis is positive for ascending)
        
        current_altitude = current_position[2]  #state.kinematics_estimated.position.z_val
       
       
        print(f"Current speed: V_x:{V_cmd_x},V_y:{V_cmd_y}")
        # Move the drone using the calculated velocity commands
        # yaw_angle = (np.arctan2(V_cmd_y,V_cmd_x))
        # yaw_angle = np.float64(np.degrees(yaw_angle))

        # client.moveByVelocityAsync(V_cmd_x,V_cmd_y,0,duration=1, drivetrain= DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(is_rate=False, yaw_or_rate=yaw_angle))
        
        '''logging velocity changing'''
        # V_cmd_x_list.append(V_cmd_x)
        # V_cmd_y_list.append(V_cmd_y)
        # V_tgt_x_list.append(V_tgt_x)
        # V_tgt_y_list.append(V_tgt_y)

        ### moving commands:
        # client.moveByVelocityAsync(V_cmd_x,V_cmd_y,0,duration=1, drivetrain= DrivetrainType.ForwardOnly)
        client.moveByVelocityZAsync(V_cmd_x,V_cmd_y,-3,duration=1, drivetrain= DrivetrainType.ForwardOnly)

        time.sleep(0.9)
        #print(f"current altitude:{current_altitude}")
        # if current_altitude > -2.9 or current_altitude < -3.1:
        #    client.moveToZAsync(-3, 1.5)
        # if current_altitude > target_z + 0.1 or current_altitude < target_z - 0.1:
        #     client.moveToZAsync(target_z, 1.5)
        
        # distance_to_target = np.sqrt((current_position[0] - target_x)**2 + (current_position[1] - target_y)**2)
        # if distance_to_target <= TARGET_REACH_THRESHOLD:
        #     print(f"Target reached at position: x={current_position[0]}, y={current_position[1]}")


        return V_cmd_x, V_cmd_y



def plot_trajectory(x_trajectory, y_trajectory,obstacle_positions, obstacle_size=0.5): #
    """Plot the recorded trajectory in 2D (x-y coordinates)."""
    plt.figure(figsize=(8, 6))
    plt.plot( y_trajectory, x_trajectory, marker='o', color='b', linestyle='-', label='Trajectory')
    #  # Plot obstacles as red 'X'
    # for obstacle in obstacle_positions:
    #     plt.scatter(obstacle[0], obstacle[1], color='r', marker='x', s=100, label='Obstacle')
    
    # for (ox, oy) in obstacle_positions:
    # # Create a square centered at (ox, oy)
    #     obstacle = patches.Rectangle((ox - obstacle_size/2, oy - obstacle_size/2), 
    #                                 obstacle_size, obstacle_size, linewidth=1, 
    #                                 edgecolor='r', facecolor='none') #, label='Obstacle'
    #     plt.gca().add_patch(obstacle)

    for obstacle in tunnel_obstacles:
        plt.scatter(obstacle[0], obstacle[1], color='k', marker='o', s=100, label='tunnel obstacle')

    # Wall positions
    wall_1_x = -3.73
    wall_2_x = 3.97
    y_min, y_max = -10, 50  # Range of y values for walls
    wall_1 = patches.Rectangle((wall_1_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)
    wall_2 = patches.Rectangle((wall_2_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)

    # Add the walls to the plot
    plt.gca().add_patch(wall_1)
    plt.gca().add_patch(wall_2)

    plt.ylim(-5,20)
    plt.xlim(-10,10)
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.title('Quadrotor 2D Trajectory (X-Y)')
    plt.legend()
    plt.grid(True)
    plt.show()

def data_logging(time_steps, ax1, ax2):

    # Ensure all lists have the same length
    if len(time_steps) != len(V_cmd_x_list) or len(time_steps) != len(V_tgt_x_list):
        print("Data lengths are inconsistent. Skipping plot update.")
        return

    # Clear previous data on the plot for real-time update
    ax1.clear()
    ax2.clear()

    # Plot V_cmd and V_tgt velocities in x and y directions
    ax1.plot(time_steps, V_cmd_x_list, label="V_cmd_x", linestyle='-') #, marker='o'
    ax1.plot(time_steps, V_tgt_x_list, label="V_tgt_x", linestyle='--') #, marker='x'
    ax1.set_ylabel("Velocity X")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(time_steps, V_cmd_y_list, label="V_cmd_y", linestyle='-') #, marker='o'
    ax2.plot(time_steps, V_tgt_y_list, label="V_tgt_y", linestyle='--') #, marker='x'
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
    V_cmd_x = None
    V_cmd_y = None
    
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
 
    while True:  # Limit the loop to 100 frames to avoid performance issues
        frame_start = time.time()
        curr_depth_images = capture_images_from_all_cameras(client)
        curr_segments = segment_image(curr_depth_images)
        if frame_count > 1:
            loom_data, flow_data = process_segments(client, curr_segments, V_cmd_x, V_cmd_y)

            if len(loom_data) == 6 and len(flow_data) == 6:
                V_cmd_x, V_cmd_y = adjust_drone_path(client, loom_data, flow_data) # V_x, V_y = velocity_controller(loom_data, flow_data, V_x, V_y)
                # client.moveByVelocityAsync(V_x, V_y, 0, duration=1)


                current_time = time.time() - start_time  # Calculate elapsed time in seconds
                time_steps.append(current_time)

                # Log and plot the velocities in real-time
                # data_logging(time_steps, ax1, ax2)
        
            else:
                print("Incomplete data; skipping frame.")
            
              
        
        curr_position = client.simGetVehiclePose().position  # Get current position
        distance_to_target = math.sqrt((curr_position.x_val - target_x)**2 + 
                                       (curr_position.y_val - target_y)**2)
         # Check if the target is reached based on distance
        if distance_to_target < TARGET_REACH_THRESHOLD:
            print("Target reached. Exiting loop.")
            break  # Exit the loop once the target is reached


        # Only visualize optic flow for the front camera
       
        # visualized_image = visualize_optic_flow(segments['front'].copy(), flow_data)
        
        # cv2.imshow("Optic Flow Visualization", visualized_image)
        # cv2.waitKey(1)

       
        frame_count += 1
        print(f"FRAME: {frame_count}")
        frame_end = time.time()
        frame_rate = 1/(frame_end-frame_start)
        print("frame rate:", frame_rate)


    average_speed_x = sum(V_cmd_x_list)/len(V_cmd_x_list)
    average_speed_y = sum(V_cmd_y_list)/len(V_cmd_y_list)

    print(f"Average speed: Vx = {average_speed_x}, Vy = {average_speed_y}")

   
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    plot_trajectory(x_trajectory, y_trajectory,obstacle_positions, obstacle_size=1)
    # plot_trajectory(x_trajectory, y_trajectory)
    
    # Outside the main loop, at the end of the program:
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plo

   
if __name__ == "__main__":
    main()
    