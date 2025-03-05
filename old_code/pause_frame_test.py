import csv
import math
import os
import airsim
import time
from statistics import mean

import cv2
from matplotlib import pyplot as plt
import numpy as np 

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff 
print("Taking off...")
client.takeoffAsync().join()
client.takeoffAsync().join()
start_position = client.getMultirotorState().kinematics_estimated.position.x_val

# Ensure the drone reaches a safe altitude before proceeding
state = client.getMultirotorState()
while state.kinematics_estimated.position.z_val > -1:
    state = client.getMultirotorState()

print("Takeoff complete. Starting forward movement...")


def draw_full_flow_arrows(flow, frame, offset_y=240 // 2 - 20 // 2, scale=50, step_x=20, step_y=2):

    H, W = flow.shape[:2]
    overlay = frame.copy()

    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            dx, dy = flow[y, x]
            # Starting point in the overlay
            start_x, start_y = x, y + offset_y
            # End point after scaling
            end_x = int(start_x + dx * scale)
            end_y = int(start_y + dy * scale)

            cv2.arrowedLine(
                overlay,
                (start_x, start_y),
                (end_x, end_y),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return overlay



def process_optical_flow(farneback_optic_flow, frame):

    farneback_flow_ds = farneback_optic_flow  
    
    # Compute pixel displacement (how many pixels moved) using the flow vector magnitudes
    flow_x = farneback_flow_ds[..., 0]
    flow_y = farneback_flow_ds[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    avg_movement = np.mean(magnitude)
    max_movement = np.max(magnitude)
    print(f"Avg movement: {avg_movement:.2f} pixels, Max movement: {max_movement:.2f} pixels")
    
    # Draw arrow visualization based on the farneback optical flow
    arrow_vis = draw_full_flow_arrows(farneback_flow_ds, frame, scale=10) #draw_farneback_arrows
    
   
    
    return arrow_vis

def OF_cal(prev_img, curr_img, prev_time, dt_sim):
    
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    

    img_cropped_height = 20  # Desired height for the middle part
    img_middle_start = 240 // 2 - img_cropped_height // 2
    img_middle_end = 240 // 2 + img_cropped_height // 2
    cropped_prev_gray = prev_gray[img_middle_start:img_middle_end, :]
    cropped_curr_gray = curr_gray[img_middle_start:img_middle_end, :]
    # 1. Calculate Farneback optical flow
    farneback_flow = cv2.calcOpticalFlowFarneback(cropped_prev_gray, cropped_curr_gray, None,
                                                  pyr_scale=0.5, levels=10, winsize=30,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # if dt_sim > 0:
    #     farneback_flow /= dt_sim 
    # Get actual dimensions of the farneback_flow array
    flow_height, flow_width = farneback_flow.shape[:2]

    # print(f"farneback flow height:{flow_height},width:{flow_width}")
    cropped_height = 6  # Desired height for the middle part
    middle_start = flow_height // 2 - cropped_height // 2
    middle_end = flow_height // 2 + cropped_height // 2
    cropped_flow = farneback_flow[middle_start:middle_end, :]

#     # Downsample flow fields for clarity
    farneback_flow_ds = cropped_flow[::1, ::1]
    

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
 
    # Retrieve depth map from AirSim (each element is the Z depth for that pixel)

    depth_image = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True)])[0]
    

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
    depth_map = depth_map[img_middle_start:img_middle_end, :]
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
    # Multiply by frame_time to scale
    geometric_flow = np.stack((Qu,Qv), axis= -1) 

    cropped_geo_flow = geometric_flow[middle_start:middle_end, :]
    # Downsample the geometric flow if needed
    geometric_flow_ds = cropped_geo_flow[::1, ::1]

    # Print dimensions of geometric_flow_ds
   # print("Dimensions of geometric_flow_ds:", geometric_flow_ds.shape)

    # Print dimensions of farneback_flow_ds
   # print("Dimensions of farneback_flow_ds:", farneback_flow_ds.shape)
    
    flow_x = farneback_flow_ds[..., 0] /Fx

    avg_flow_x = np.mean(flow_x, axis=0)
    avg_flow_x = np.abs(avg_flow_x)
   
    cal_flow_x = geometric_flow_ds[..., 0] /Fx
    avg_cal_flow_x = np.mean(cal_flow_x, axis=0)
    avg_cal_flow_x = np.abs(avg_cal_flow_x)


    row2 = avg_cal_flow_x.tolist()  
    row1 = [x * 1/dt_sim for x in avg_flow_x]  
    all_avg_flow_x.append(row1[:]) 
    all_avg_cal_flow_x.append(row2[:])  
    print(f"frame rate: {dt_sim} seconds for 1 frame")

    return dt, farneback_flow_ds
    # row1 = avg_flow_x.tolist()
    # writer.writerow(row1)
    # file.flush()  # Ensure data is saved immediately
    # row2 = avg_cal_flow_x.tolist() 
    # curr_time = time.time()
    # elapsed = curr_time - frame_start
    # row2 = [x * elapsed for x in row2]
    # writer.writerow(row2)
    # file.flush()


camera_view_folder_path=r"saved_camera_views"
os.makedirs(camera_view_folder_path, exist_ok=True)
velocity = 1  # m/s
# duration = 5  # seconds
target_distance = 5 # meters
frame_count = 0
image_data = []  # Store images in memory
image_count = 0
output_csv = 'average_optical_flow.csv'
all_avg_flow_x = []
all_avg_cal_flow_x = []
V_x_list = []
V_y_list = []
position_x_list = []

def data_logging(time_steps, ax1):
    if len(time_steps) != len(V_x_list):
        print("Data lengths are inconsistent. Skip")
        return

    # Clear previous data on the plot for real-time update
    ax1.clear()

    # Plot V_cmd and V_tgt velocities in x and y directions
    ax1.plot(time_steps, V_x_list, label="V_x", linestyle='-', marker='o')
    ax1.set_ylabel("Velocity X")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    plt.pause(0.01)

def position_drawing(time_steps_1, x_position_ax):
    if len(time_steps_1) != len(position_x_list):
        print("Data lengths are inconsistent. Skip")
        return
    x_position_ax.clear()
    
    x_position_ax.plot(time_steps_1, position_x_list, label="Position_x", linestyle = '-', marker='x')
    x_position_ax.set_ylabel("Position x")
    x_position_ax.grid(True)
    plt.pause(0.01)

# Move forwardand ensure stability
# client.moveByVelocityZAsync(velocity, 0, 0, duration)
fig, ax1 = plt.subplots()  
fig2, x_position_ax = plt.subplots()
time_steps = []
time_steps_1 = []
dt = 0
desired_frame_interval = 0.018  # 18ms per frame (55 FPS)   # 20ms per frame (50 FPS)
start_time = time.time()
client.simPause(True)  # Pause simulation
last_frame_time = client.getMultirotorState().timestamp
start_sim_time = last_frame_time

while True:
    
    current_position = client.getMultirotorState().kinematics_estimated.position.x_val
    distance_traveled = current_position - start_position

    # Check if the drone has moved 5 meters forward
    if distance_traveled >= target_distance:
        break 

    state_prev = client.getMultirotorState()
    prev_time = state_prev.timestamp
    
    
    # client.simContinueForTime(0.035)

    # client.moveByVelocityZAsync(velocity, 0, -5, 1)
    client.moveByVelocityAsync(velocity, 0, 0, 1)
    client.simContinueForTime(desired_frame_interval)  # Now 0.02s to match FPS
    # # Get the drone's velocity !! before pausing check point !!
    # state = client.getMultirotorState()

    # V_x = state.kinematics_estimated.linear_velocity.x_val   
    # V_x_list.append(V_x)
    
    # current_time = time.time() - start_time
    # time_steps.append(current_time)
    # data_logging(time_steps, ax1)

   

    # Pause before capturing images
    client.simPause(True)

    current_sim_time = client.getMultirotorState().timestamp
   
    time_since_last_frame = (current_sim_time - last_frame_time) / 1e9  # Convert nanoseconds to seconds
    # print(f"before cap time interval in simulation: {time_since_last_frame:.6f}")
    # Pause the simulation
    
    # client.simContinueForTime(0.2)

    responses = client.simGetImages([
        airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)  
        # airsim.ImageRequest("left", airsim.ImageType.Scene, False, False), 
        # airsim.ImageRequest("right", airsim.ImageType.Scene, False, False), 
        # airsim.ImageRequest("bottom", airsim.ImageType.Scene, False, False)
    ])
    curr_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)

    for response in responses:
        if response.image_data_uint8:
            image_count += 1


    state_next = client.getMultirotorState()
    next_time = state_next.timestamp  # Get new sim time

    dt_sim = (next_time - prev_time) / 1e9  # Convert nanoseconds to seconds
    curr_time = time.time()
    im_path = os.path.join(camera_view_folder_path,f'{frame_count}.png')
    if frame_count >=1:
        dt, farneback_OF = OF_cal(prev_img, curr_img, prev_time, dt_sim)
        # prev_time = time.time()
        # position_x = state_next.kinematics_estimated.position.x_val
        # position_x_list.append(position_x)
        # current_time = time.time() - start_time
        # time_steps_1.append(current_time)
        # position_drawing(time_steps_1, x_position_ax)
        arrow_vis = process_optical_flow(farneback_OF, curr_img)
        cv2.imshow("Farneback Optical Flow", arrow_vis)
        cv2.waitKey(30)


    # print(f"dt used for Farneback (simulated time): {dt_sim:.6f} seconds")

    cv2.imwrite(im_path, curr_img)
    # current_time = time.time() - start_time  # Calculate elapsed time in seconds
    # state = client.getMultirotorState()
    # V_x = state.kinematics_estimated.linear_velocity.x_val
    # V_x_list.append(V_x)
    # time_steps.append(current_time)
    # data_logging(time_steps, ax1)
    frame_count += 1
    prev_img = curr_img

    last_frame_time = current_sim_time
    
    # time.sleep(0.08)  # Allow rendering before next pause

client.simPause(False)  # Resume simulation
end_time = time.time()
end_sim_time = client.getMultirotorState().timestamp
real_time_dur = end_time - start_time
sim_time_dur = (end_sim_time - start_sim_time) / 1e9 
print("Duration in real time:", real_time_dur)
print("Duration in simulation time:", sim_time_dur)
# client.landAsync().join()
client.enableApiControl(False)
client.armDisarm(False)

print(f"Total images captured: {image_count}")

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row1, row2 in zip(all_avg_flow_x, all_avg_cal_flow_x):
        writer.writerow(row1)
        writer.writerow(row2)

print("Script finished successfully.")
