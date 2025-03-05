import airsim
import cv2
import numpy as np
import time
import random

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off and hover at 5 meters
client.takeoffAsync().join()
client.moveToZAsync(-5, 1).join()
time.sleep(2)
print("X Position:", client.getMultirotorState().kinematics_estimated.position.x_val)
print("Y Position:", client.getMultirotorState().kinematics_estimated.position.y_val)

#  get grayscale image from downward camera
def get_camera_image():
    responses = client.simGetImages([airsim.ImageRequest("bottom", airsim.ImageType.Scene)])
    if responses and len(responses) > 0:
        response = responses[0]
        if response.width == 0 or response.height == 0:
            return None
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return cv2.imdecode(img1d, cv2.IMREAD_GRAYSCALE)
    return None

# Capture the initial reference image
prev_img = get_camera_image()
if prev_img is None:
    print("Error: Could not capture reference image.")
    exit(1)

# Apply an initial drift to test the damping effect
client.moveByVelocityAsync(1, 0, 0, 1).join()  # Move forward for 2 sec

# Control parameters
kp = 1.5  # Increased gain to see movement
scale_factor = 0.03  
dt = 0.02  # Faster loop
damping_factor = 0.9  # Reduces excessive movement over time

# Function to visualize optical flow
def draw_optical_flow(img, flow):
    h, w = img.shape[:2]
    step = 10
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx * 5, y + fy * 5]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis

yaw_interval = 5
last_yaw_time = time.time()
current_yaw = 0

# Start hover loop
start_time = time.time()
while time.time() - start_time < 60:
    cur_img = get_camera_image()
    if cur_img is None:
        continue

    # Compute optical flow with better sensitivity
    flow = cv2.calcOpticalFlowFarneback(prev_img, cur_img, None, 
                                        pyr_scale=0.5, levels=5, winsize=25, 
                                        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)

    # Compute average motion
    flow_x = np.mean(flow[..., 0])  # Motion along X-axis
    flow_y = np.mean(flow[..., 1])  # Motion along Y-axis

    # Convert pixel movement to real-world displacement
    error_forward = flow_y * scale_factor
    error_right = flow_x * scale_factor

    # Compute corrective velocities
    vx = -kp * error_forward
    vy = -kp * error_right

     # Apply damping to avoid excessive drift
    vx *= damping_factor
    vy *= damping_factor


    if time.time() - last_yaw_time > yaw_interval:
        yaw_change = random.uniform(-60, 60)  # Random yaw shift between -30° and 30°
        current_yaw += yaw_change
        current_yaw = current_yaw % 360  # Keep yaw within 0-360 range
        print(f"Performing Yaw Saccade to {current_yaw:.1f}°")

        # Perform yaw rotation in place
        client.rotateToYawAsync(current_yaw, 10).join()

        # Reset yaw timer
        last_yaw_time = time.time()
    

    client.moveByVelocityZAsync(vx, vy, -5, dt)

    # Visualize Optical Flow (for debugging)
    flow_visual = draw_optical_flow(cur_img, flow)
    cv2.imshow("Optical Flow", flow_visual)
    cv2.waitKey(1)


    print(f"Flow: ({flow_x:.2f}, {flow_y:.2f}), vx={vx:.2f}, vy={vy:.2f}")

    # Update previous image
    prev_img = cur_img

    time.sleep(dt)

print("X Position:", client.getMultirotorState().kinematics_estimated.position.x_val)
print("Y Position:", client.getMultirotorState().kinematics_estimated.position.y_val)
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()
