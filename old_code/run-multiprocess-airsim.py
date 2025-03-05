import math
import airsim
import time
import numpy as np
import cv2
import os
import csv
from multiprocessing import Process, Queue
from scipy.signal import butter, filtfilt

# Function for optical flow calculation
def OF_cal_worker(calculation_queue, result_queue):
    image_width=320
    image_height=240
    all_avg_flow_x_l = []
    all_avg_cal_flow_x_l = []
    # Create directory for saving images
    # camera_view_folder_path = r"saved_camera_views_single"
    # os.makedirs(camera_view_folder_path, exist_ok=True)
    def moving_average_filter(signal, window_size=10):
    # symmetric filtering
        if window_size % 2 == 0:
            window_size += 1
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
  
    while True:
        data = calculation_queue.get()
        if data is None:
            break

        prev_img, curr_img, kinematics, depth_image,frame_start,fov = data

        cropped_height = 100  # Desired height for the middle part
        middle_start = image_height // 2 - cropped_height // 2
        middle_end = image_height // 2 + cropped_height // 2
        prev_img= prev_img[middle_start:middle_end, :]
        curr_img= curr_img[middle_start:middle_end, :]
        # Convert images to grayscale
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # # Save the current image
        # curr_time = time.time()
        # im_path = os.path.join(camera_view_folder_path, f'{curr_time}_current.png')
        # cv2.imwrite(im_path, curr_img)


        # 1. Calculate Farneback optical flow
        farneback_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                    pyr_scale=0.5, levels=5, winsize=30,
                                                    iterations=5, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


        # Get actual dimensions of the farneback_flow array
        flow_height, flow_width = farneback_flow.shape[:2]

        # print(f"farneback flow height:{flow_height},width:{flow_width}")
        cropped_height = 10  # Desired height for the middle part
        middle_start = flow_height // 2 - cropped_height // 2
        middle_end = flow_height // 2 + cropped_height // 2
        cropped_flow = farneback_flow[middle_start:middle_end, :]

    #     # Downsample flow fields for clarity
        farneback_flow_ds = cropped_flow[::1, ::1]

        # 2. Calculate Geometric Optical Flow

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
    
        # Convert depth data to a numpy array and reshape it to the image dimensions
        depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)

        focal_length_x = image_width / (2 * math.tan(fov / 2))

        vertical_FOV = 2 * math.atan((math.tan(fov/2)) / (image_width/image_height))
        focal_length_y = image_height / (2 * math.tan(vertical_FOV / 2))
        # Assuming these are already defined in your code
        Fx = focal_length_x # focal length in pixels (Horizontal = Vertical)
        Fy = focal_length_y
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
        
        flow_x = farneback_flow_ds[..., 0] 

        avg_flow_x = np.mean(flow_x, axis=0)
        avg_flow_x = np.abs(avg_flow_x)/Fx
        avg_flow_x = moving_average_filter(avg_flow_x, window_size=20)
        
        cal_flow_x = geometric_flow_ds[..., 0] 
        avg_cal_flow_x = np.mean(cal_flow_x, axis=0)
        avg_cal_flow_x = np.abs(avg_cal_flow_x)/Fx


        row2 = avg_cal_flow_x.tolist()  
        curr_time = time.time()
        elapsed = curr_time - frame_start
        row1 = [x * 1/elapsed for x in avg_flow_x]  
        all_avg_flow_x_l.append(row1[:]) 
        all_avg_cal_flow_x_l.append(row2[:])
        result_queue.put((row1, row2))  # Send a copy


def controller(flow_x):   
    gain_x_1 = 1
    gain_x_2 = 0.1
    T = 1/24
    # Acce_x = (sum(flow_x[-1])-T) * gain_x_1 + (flow_x[-2] - flow_x[-1])* gain_x_2 # - dragx
    Acce_x = (sum(flow_x[-1]) - T) * gain_x_1 + np.array(flow_x[-2]) - np.array(flow_x[-1]) * gain_x_2
    # Set initial conditions
    return Acce_x
       


fov=90
# Initialize AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.simSetCameraFov("front", fov)
client.takeoffAsync().join()
client.takeoffAsync().join()
        

# Setup multiprocessing
calculation_queue = Queue()
result_queue = Queue()
process = Process(target=OF_cal_worker, args=(calculation_queue, result_queue))
process.start()

# Variables for storing results
all_avg_flow_x = []
all_avg_cal_flow_x = []


# Start drone movement
start_time = time.time()
frame_count = 0

speed=1
client.moveByVelocityAsync(speed, 0, 0, 10)
while time.time() - start_time < 5:
    frame_start = time.time()
    # Capture the current frame
    image = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])[0]
    curr_img = np.frombuffer(image.image_data_uint8, dtype=np.uint8).reshape(image.height, image.width, 3)

  
    if frame_count > 0:
        # Retrieve depth and kinematics data
        depth_image = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPerspective, True)])[0]
        kinematics = client.simGetGroundTruthKinematics()

        # Send data to the worker process
        calculation_queue.put((prev_img, curr_img, kinematics, depth_image,frame_start,fov))
    # Drain remaining results from result_queue
    while not result_queue.empty():
        k, l = result_queue.get()
        all_avg_flow_x.append(k)
        all_avg_cal_flow_x.append(l)
        ###control things happen
    # Retrieve results from the worker process
    if len(all_avg_flow_x) > 2:
        controller(all_avg_flow_x)
    prev_img = curr_img
    frame_count += 1

    # Log frame processing time
    elapsed = time.time() - frame_start
    print(f"Frame rate: {1 / elapsed:.2f} FPS")


# Stop the quadroto

client.hoverAsync().join()
time.sleep(1)
#client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

calculation_queue.put(None)

# Wait for the worker to finish
print("Waiting for worker process to terminate...")
process.join(timeout=1)

# Force termination if still alive
if process.is_alive():
    print("Worker process did not terminate, forcing termination.")
    process.terminate()
    process.join()

# Save results to CSV
output_csv = 'average_optical_flow.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row1, row2 in zip(all_avg_flow_x,all_avg_cal_flow_x):
        writer.writerow(row1)
        writer.writerow(row2)
