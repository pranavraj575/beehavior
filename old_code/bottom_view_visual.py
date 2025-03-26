import csv
import airsim
import numpy as np
import cv2
import time
import random
from airsim.types import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation



client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

def capture_image(client, camera_name):
    """Capture a single image from the specified camera and downsample it."""
    try:
        response = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
        if response:
            img_data = response[0].image_data_uint8
            if img_data:
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
                #image = cv2.resize(image, (320, 240))  #(1080, 720) : Resize to 320x240 for performance
                print(f"Captured {camera_name} image, shape: {image.shape}")
                return image
            else:
                print(f"Warning: Empty image data from {camera_name}")
        else:
            print(f"Warning: No response from {camera_name}")
    except Exception as e:
        print(f"Error capturing image from {camera_name}: {e}")
    return None

def calculate_optic_flow(prev_image, curr_image):
    """Calculates the dense optic flow between two images with optimized settings."""

    # def moving_average_filter(signal, window_size=10):
    # # Ensure the window size is odd for symmetric filtering
    #     if window_size % 2 == 0:
    #         window_size += 1
    #     return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
  
    # Convert the images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

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
        levels=3,
        winsize=15,
        iterations=5,
        poly_n=5,
        poly_sigma=1.1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    # flow = moving_average_filter(flow, window_size=20)
    return flow
 

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


def visualize_optic_flow(image, flow, local_size=50, step=32, direction_threshold=15):
    
    h, w = flow.shape[:2]
    half_local_size = local_size // 2

    # Iterate over the grid positions
    for y in range(half_local_size, h, step):
        for x in range(half_local_size, w, step):
            # Define the local region for averaging
            y_start, y_end = max(0, y - half_local_size), min(h, y + half_local_size)
            x_start, x_end = max(0, x - half_local_size), min(w, x + half_local_size)
            
            # Extract local flow
            local_flow = flow[y_start:y_end, x_start:x_end]
            
            # Calculate average flow in the local region
            avg_fx = np.mean(local_flow[..., 0])
            avg_fy = np.mean(local_flow[..., 1])
            
            # Calculate the magnitude and angle
            magnitude = np.linalg.norm([avg_fx, avg_fy])
            angle = np.degrees(np.arctan2(avg_fy, avg_fx))

            # Filter: Ensure consistent direction and magnitude with nearby arrows
            if magnitude > 1e-2:  # Ignore insignificant flow
                # Scale the flow for visualization
                start_x, start_y = x, y
                end_x = int(start_x + avg_fx * 10)
                end_y = int(start_y + avg_fy * 10)

                # Draw the arrow on the image
                cv2.arrowedLine(image, (start_x, start_y+324), (end_x, end_y+324), (0, 0, 255), 2, tipLength=0.5)

    return image



def OF_cal(prev_img, curr_img):
    def moving_average_filter(signal, window_size=10):
    # symmetric filtering
        if window_size % 2 == 0:
            window_size += 1
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
  
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # Crop to the middle part of the height
    h, w = prev_gray.shape
    cropped_height = 72  # Desired height for the middle part
    middle_start = h // 2 - cropped_height // 2
    middle_end = h // 2 + cropped_height // 2
    prev_gray_cropped = prev_gray[middle_start:middle_end, :]
    curr_gray_cropped = curr_gray[middle_start:middle_end, :]
    # 1. Calculate Farneback optical flow
    farneback_flow = cv2.calcOpticalFlowFarneback(prev_gray_cropped, curr_gray_cropped, None,
                                                  pyr_scale=0.5, levels=5, winsize=30,
                                                  iterations=5, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    

    # Get actual dimensions of the farneback_flow array
    flow_height, flow_width = farneback_flow.shape[:2]

    # print(f"farneback flow height:{flow_height},width:{flow_width}")
    cropped_height = 20  # Desired height for the middle part
    middle_start = flow_height // 2 - cropped_height // 2
    middle_end = flow_height // 2 + cropped_height // 2
    cropped_flow = farneback_flow[middle_start:middle_end, :]

#     # Downsample flow fields for clarity
    farneback_flow_ds = cropped_flow[::1, ::1]
    

    # 2. Calculate Geometric Optical Flow
    # Retrieve real kinematic data from AirSim
    kinematics = client.simGetGroundTruthKinematics()

    '''Have changed the coordinate transformation for downwards camera'''
    # Translational velocity (linear velocity) !!! switch x,y,z velocity due to Airsim setting
    T = np.array([   
        kinematics.linear_velocity.y_val,
        kinematics.linear_velocity.x_val,  
        -kinematics.linear_velocity.z_val
        
    ])

    # Rotational velocity (angular velocity)
    omega = np.array([
        
        kinematics.angular_velocity.y_val,
        kinematics.angular_velocity.x_val,
        -kinematics.angular_velocity.z_val
        
    ])
 
    # Retrieve depth map from AirSim (each element is the Z depth for that pixel)

    depth_image = client.simGetImages([airsim.ImageRequest("bottom", airsim.ImageType.DepthPerspective, True)])[0]
    

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
    
    # # Print dimensions to verify
    # print(f"Scene Image Shape: {curr_img.shape}")
    print(f"Depth Image Shape: {depth_map.shape}")


    # Assuming these are already defined in your code
    Fx = Fy = 160.2694 # focal length in pixels (Horizontal = Vertical)
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
# avg_flow_x_g = butter_lowpass_filter(avg_flow_x, cutoff_frequency=0.3, fs=1, order=3)

    cropped_geo_flow = geometric_flow[middle_start:middle_end, :]
    # Downsample the geometric flow if needed
    geometric_flow_ds = cropped_geo_flow[::1, ::1]

    # Print dimensions of geometric_flow_ds
   # print("Dimensions of geometric_flow_ds:", geometric_flow_ds.shape)

    # Print dimensions of farneback_flow_ds
   # print("Dimensions of farneback_flow_ds:", farneback_flow_ds.shape)
    
    flow_x = farneback_flow_ds[..., 1] /Fx

    avg_flow_x = np.mean(flow_x, axis=0)
    # avg_flow_x = np.abs(avg_flow_x)
    avg_flow_x = moving_average_filter(avg_flow_x, window_size=20)
      
   
    cal_flow_x = geometric_flow_ds[..., 1] /Fx
    avg_cal_flow_x = np.mean(cal_flow_x, axis=0)
    # avg_cal_flow_x = np.abs(avg_cal_flow_x)


    row2 = avg_cal_flow_x.tolist()  
    curr_time = time.time()
    elapsed = curr_time - frame_start
    row1 = [x * 1/elapsed for x in avg_flow_x]  
    all_avg_flow_x.append(row1[:]) 
    all_avg_cal_flow_x.append(row2[:])  
    print(f"frame rate: {elapsed} seconds for 1 frame")

    # row1 = avg_flow_x.tolist()
    # writer.writerow(row1)
    # file.flush()  # Ensure data is saved immediately
    # row2 = avg_cal_flow_x.tolist() 
    # curr_time = time.time()
    # elapsed = curr_time - frame_start
    # row2 = [x * elapsed for x in row2]
    # writer.writerow(row2)
    # file.flush()


camera_info = client.simGetCameraInfo("bottom")  # Replace "0" with your camera ID if different

# Field of View (in radians) and image width
fov = math.radians(camera_info.fov)  # Convert FoV to radians
print("camera_info:", camera_info)
print("filed of view (detected):", fov)

frame_count = 0
output_csv = 'average_optical_flow.csv'
all_avg_flow_x = []
all_avg_cal_flow_x = []
start_time = time.time()
client.moveByVelocityAsync(0, 0, 0, 10)
# yaw_rate_deg_per_sec = 1* (180 / 3.141592653589793)  # Convert 2 rad/s to degrees/s
# client.rotateByYawRateAsync(yaw_rate_deg_per_sec, 10)
   
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)  # <-- CORRECT: Create a CSV writer object
while time.time() - start_time < 5:
    frame_start = time.time()
# Capture two consecutive images for optical flow calculation
    curr_img =  capture_image(client, "front")

    if frame_count >= 1:
        # flow = calculate_optic_flow(prev_img, curr_img)
        # avg_flow = calculate_average_flow(flow)
        # flow = OF_cal(prev_img, curr_img)
        if frame_count % 10 == 0:
            #visualized_image = visualize_optic_flow(curr_img.copy(), flow) #visualize_optic_flow(curr_images[camera_name].copy(), flow)
            cv2.imshow("Optic Flow Visualization", curr_img)
            cv2.waitKey(1)
    

    
    curr_time = time.time()

    
 
    frame_count += 1
    prev_img = curr_img
    print("frame rate", 1/(curr_time - frame_start))

client.hoverAsync().join()
client.armDisarm(False)
client.enableApiControl(False)


with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row1, row2 in zip(all_avg_flow_x, all_avg_cal_flow_x):
        writer.writerow(row1)
        writer.writerow(row2)