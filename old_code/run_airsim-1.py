import airsim
import time
import numpy as np
import cv2
import os
import csv

# target_fps = 30
# capture_interval = 1.0 / target_fps


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

def OF_cal(prev_img, curr_img):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("prev_gray",prev_gray)
    # cv2.waitKey(500)
    # cv2.imshow("curr_gray",curr_gray)
    # curr_time = time.time()
    # camera_view_folder_path=r"saved_camera_views"
    # prev_im_path = os.path.join(camera_view_folder_path,f'{curr_time}_previous.png')
    # cur_im_path = os.path.join(camera_view_folder_path,f'{curr_time}_current.png')
    # # test_im_path = os.path.join(camera_view_folder_path,f'{curr_time}_test.png')
    # cv2.imwrite(prev_im_path, prev_img)
    # cv2.imwrite(cur_im_path, curr_img)

    # 1. Calculate Farneback optical flow
    farneback_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                  pyr_scale=0.5, levels=3, winsize=9,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
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

    depth_image = client.simGetImages([airsim.ImageRequest("front1", airsim.ImageType.DepthPerspective, True)])[0]
    

    # Convert depth data to a numpy array and reshape it to the image dimensions
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)
    
    # # Print dimensions to verify
    # print(f"Scene Image Shape: {curr_img.shape}")
    # print(f"Depth Image Shape: {depth_map.shape}")


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



camera_view_folder_path=r"saved_camera_views_single"
os.makedirs(camera_view_folder_path, exist_ok=True)
frame_count = 0
output_csv = 'average_optical_flow.csv'
all_avg_flow_x = []
all_avg_cal_flow_x = []
start_time = time.time()
client.moveByVelocityAsync(1, 0, 0, 10)  
# yaw_rate_deg_per_sec = 1* (180 / 3.141592653589793)  # Convert 2 rad/s to degrees/s
# client.rotateByYawRateAsync(yaw_rate_deg_per_sec, 10)
   
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)  # <-- CORRECT: Create a CSV writer object
while time.time() - start_time < 5:
    frame_start = time.time()
# Capture two consecutive images for optical flow calculation
    image = client.simGetImages([airsim.ImageRequest("front1", airsim.ImageType.Scene, False, False)])[0]
    curr_img = np.frombuffer(image.image_data_uint8, dtype=np.uint8).reshape(image.height, image.width, 3)
    
    curr_time = time.time()

    im_path = os.path.join(camera_view_folder_path,f'{curr_time}_previous.png')
    
    cv2.imwrite(im_path, curr_img)

    if frame_count >=1:
        OF_cal(prev_img, curr_img)
        # avg_flow_x, avg_flow_y =  data_logging()
    
    frame_count += 1
    prev_img = curr_img
    
        #print(f"frame rate: {elapsed} seconds for 1 frame")
        # time.sleep(max(0, capture_interval - elapsed))  # Enforce targ


# Stop the quadrotor
client.hoverAsync().join()
time.sleep(5)
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)


with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row1, row2 in zip(all_avg_flow_x, all_avg_cal_flow_x):
        writer.writerow(row1)
        writer.writerow(row2)


