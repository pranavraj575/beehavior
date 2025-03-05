import airsim
import numpy as np
import cv2
from airsim.types import *

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToZAsync(-3,1.5).join()
client.simPause(True)

camera_name = "front"


response = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])

img_data = response[0].image_data_uint8

image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
prev_image = cv2.resize(image, (320, 240))  # (1080, 720): Resize to 320x240 for performance
print(f"Captured previous {camera_name} image, shape: {image.shape}")


client.moveByVelocityAsync(1, 0, 0, 1)
client.simContinueForTime(0.018) 
client.simPause(True)

response = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])

img_data = response[0].image_data_uint8

image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
curr_image = cv2.resize(image, (320, 240))  # (1080, 720): Resize to 320x240 for performance
print(f"Captured current {camera_name} image, shape: {image.shape}")



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

'''Do something to track flow here '''
print(f"optic flow dimension: {flow.shape}")


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

depth_image = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True)])[0]


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
# Multiply by frame_time to scale
geometric_flow = np.stack((Qu,Qv), axis= -1) 

# Downsample the geometric flow if needed
geometric_flow_ds = geometric_flow[::1, ::1]

print(f"geometirc opric flow dimension:{geometric_flow_ds.shape}")


mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
avg_mag = np.mean(mag)
print(f"Average optic flow magnitude: {avg_mag}")


client.simPause(False)
client.landAsync().join()
client.enableApiControl(False)
client.armDisarm(False)