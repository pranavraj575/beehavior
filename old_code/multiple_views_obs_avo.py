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
# starting point in tunel: (-11645.0, -22023.0, 10)

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
    camera_names = ["right", "front", "left"]
    images = {}

    for camera_name in camera_names:
        image = capture_image(client, camera_name)
        if image is not None:
            images[camera_name] = image
        else:
            print(f"Warning: Could not capture image from {camera_name}.")
    
    return images

def segment_image(images):
   
    segments = []

    for img in images.values():
        h, w, _ = img.shape
        mid_w = w // 2  # Midpoint for horizontal division

        # Divide the image into two horizontal segments
        segment_1 = img[:, :mid_w, :]
        segment_2 = img[:, mid_w:, :]

        # Add the segments to the list
        segments.extend([segment_1, segment_2])
        
    return segments

def process_segments(curr_segments, prev_segments, V_cmd_x, V_cmd_y):
  
    loom_data = []
    flow_data = []

    # Crop to the middle part of the height
    cropped_height = 10  # Desired height for the middle part
    h = 240
    middle_start = h // 2 - cropped_height // 2
    middle_end = h // 2 + cropped_height // 2

    for i in range(len(curr_segments)):
        curr_segment = curr_segments[i]
        prev_segment = prev_segments[i]

        # Convert to grayscale for optic flow calculation
        prev_gray = cv2.cvtColor(prev_segment, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_segment, cv2.COLOR_BGR2GRAY)

        
        prev_gray_cropped = prev_gray[middle_start:middle_end, :]
        curr_gray_cropped = curr_gray[middle_start:middle_end, :]


        # Calculate optic flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_cropped, curr_gray_cropped, None, 
            pyr_scale=0.5, levels=5, winsize=30, 
            iterations=5, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        # Calculate average flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_flow = np.mean(mag)  # Average magnitude as scalar

        # Calculate looming using flow magnitudes
        loom_value = calculate_loom(flow, V_cmd_x, V_cmd_y, max_loom_threshold=0.5) 

        loom_data.append(loom_value)
        flow_data.append(avg_flow)  # Store as scalar

    # Debugging output to confirm structure
    print(f"Loom data: {loom_data}")
    print(f"Flow data: {flow_data}")

    return loom_data, flow_data


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


def calculate_loom(flow, V_cmd_x, V_cmd_y, max_loom_threshold = 0.5):
    """
    Calculate loom focused on the moving direction section.
    """
     #  first frame, calculate front view loom
    if V_cmd_x is None or V_cmd_y is None:
        print("No commanded velocity provided, calculating front view loom.")
        
        # Use the entire flow to calculate average magnitude for the front view
        # mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # front_loom = np.mean(mag)
        # front_loom_clipped = np.clip(front_loom, -max_loom_threshold, max_loom_threshold)
        front_loom_clipped = loom_left_right(flow)
        print(f"Front view loom (initial frame): {front_loom_clipped}")
        return front_loom_clipped
    
    # Section centers (in radians): [pi/12, pi/4  ...]
    section_centers = np.linspace(np.pi / 12, 11 * np.pi / 12, 6)
    section_width = np.pi / 6  # 30 degrees per section

    # # Section centers (in radians): [-pi/8, pi/8,  ...]
    # section_centers = np.linspace(-np.pi / 8, 7 * np.pi / 4, 8)
    # section_width = np.pi / 4  # 45 degrees per section

    # Compute moving direction
    moving_direction = np.arctan2(V_cmd_y, V_cmd_x) % (2 * np.pi)

    # Find nearest section
    section_diffs = np.abs(section_centers - moving_direction)
    nearest_section = np.argmin(section_diffs)
    section_start = (section_centers[nearest_section] - section_width / 2) % (2 * np.pi)
    section_end = (section_centers[nearest_section] + section_width / 2) % (2 * np.pi)

    print(f"Moving Direction: {np.rad2deg(moving_direction):.2f}°")
    print(f"Nearest Section Center: {np.rad2deg(section_centers[nearest_section]):.2f}°")

    # Compute divergence of the flow field
    flow_divergence = np.zeros_like(flow[..., 0])
    height, width = flow.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Partial derivatives using finite differences
            d_fx_dx = (flow[y, x + 1, 0] - flow[y, x - 1, 0]) / 2
            d_fy_dy = (flow[y + 1, x, 1] - flow[y - 1, x, 1]) / 2
            flow_divergence[y, x] = d_fx_dx + d_fy_dy

    # Mask flow_divergence for the nearest section
    loom_value = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            fx, fy = flow[y, x]
            flow_angle = np.arctan2(fy, fx) % (2 * np.pi)
            if section_start < section_end:
                in_section = section_start <= flow_angle <= section_end
            else:
                in_section = flow_angle >= section_start or flow_angle <= section_end

            if in_section:
                loom_value += flow_divergence[y, x]  # Use divergence directly
 
    # Normalize loom value by number of pixels
    num_pixels = height * width
    loom_value /= num_pixels

    loom_value_scal = loom_value *10 

    print(f"Loom Value (Section {nearest_section}): {loom_value_scal}")
    return loom_value_scal

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

def velocity_controller(loom_data, flow_data, V_x, V_y, loom_weight = 5):
   
    # Directions: [front-left, front-right, right-front, right-back, back-right, back-left, left-back, left-front]
    num_directions = len(loom_data)
    direction_weights = [4.0] * num_directions  # Adjust weights for directional priority

    # # Normalize flow and loom data
    # norm_flow = [flow / max(flow_data) if max(flow_data) > 0 else 0 for flow in flow_data]
    # norm_loom = [loom / max(loom_data) if max(loom_data) > 0 else 0 for loom in loom_data]

    # Compute adjustments based on normalized flow and loom
    adjustments = [direction_weights[i] * (flow_data[i] + loom_weight * loom_data[i]) for i in range(num_directions)]

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


    norm = np.sqrt(V_x_adjust**2 + V_y_adjust**2)
    if norm > 0:
        V_x_adjust /= norm
        V_y_adjust /= norm

    # Combine adjustments with existing velocity
    V_cmd_x = V_x + V_x_adjust
    V_cmd_y = V_y + V_y_adjust

    return V_cmd_x, V_cmd_y


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
        V_x = state.kinematics_estimated.linear_velocity.x_val
        V_y = state.kinematics_estimated.linear_velocity.y_val
        print(f"Previous speed:V_x:{V_x},V_y:{V_y}")
        
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
        V_cmd_x_list.append(V_cmd_x)
        V_cmd_y_list.append(V_cmd_y)
        V_tgt_x_list.append(V_tgt_x)
        V_tgt_y_list.append(V_tgt_y)

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



    # Wall positions
    wall_1_x = -3.73
    wall_2_x = 3.97
    y_min, y_max = -10, 50  # Range of y values for walls
    wall_1 = patches.Rectangle((wall_1_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)
    wall_2 = patches.Rectangle((wall_2_x, y_min), 0.1, y_max - y_min, color='green', alpha=0.5)

    # Add the walls to the plot
    plt.gca().add_patch(wall_1)
    plt.gca().add_patch(wall_2)

    plt.ylim(0,20)
    plt.xlim(0,20)
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
        curr_images = capture_images_from_all_cameras(client)
        curr_segments = segment_image(curr_images)
        if frame_count > 1:
            loom_data, flow_data = process_segments(curr_segments, prev_segments, V_cmd_x, V_cmd_y)

            if len(loom_data) == 6 and len(flow_data) == 6:
                V_cmd_x, V_cmd_y = adjust_drone_path(client, loom_data, flow_data) # V_x, V_y = velocity_controller(loom_data, flow_data, V_x, V_y)
                # client.moveByVelocityAsync(V_x, V_y, 0, duration=1)


                current_time = time.time() - start_time  # Calculate elapsed time in seconds
                time_steps.append(current_time)

                # Log and plot the velocities in real-time
                data_logging(time_steps, ax1, ax2)
        
            else:
                print("Incomplete data; skipping frame.")
            
        curr_time = time.time()       
        
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

        # prev_time = curr_time  # Update time
        prev_images = curr_images
        prev_segments = curr_segments
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
    