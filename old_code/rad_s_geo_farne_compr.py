import airsim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Define initial and target positions
initial_pose = client.simGetVehiclePose().position
# target_position = initial_pose + airsim.Vector3r(10, 0, 0)  # Move 10 meters forward
# Set initial velocity and duration
velocity = 1  # Move forward at 1 m/s
duration = 60  # Move forward for 10 seconds
interval = 0.1 

# Take off to ensure the drone is airborne
client.takeoffAsync().join()

# Camera parameters
focal_length_px = 128.2155  # Example focal length in pixels
fps = 14  # 1.25 # Frames per second

# Initialize the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
geom_ax, farneback_ax, diff_ax = axes

# Create dummy initial data to initialize plots and colorbars
initial_data = np.zeros((240, 320))  #image dimensions
geom_plot = geom_ax.imshow(initial_data, cmap='jet')
farneback_plot = farneback_ax.imshow(initial_data, cmap='jet')
diff_plot = diff_ax.imshow(initial_data, cmap='jet')

# Add colorbars once
geom_cbar = fig.colorbar(geom_plot, ax=geom_ax)
farneback_cbar = fig.colorbar(farneback_plot, ax=farneback_ax)
diff_cbar = fig.colorbar(diff_plot, ax=diff_ax)

# Set titles and labels for the subplots
geom_ax.set_title("Geometric Flow (Angular Magnitude, rad/s)")
geom_ax.set_xlabel("Pixel X Coordinate")
geom_ax.set_ylabel("Pixel Y Coordinate")
geom_ax.invert_yaxis()

farneback_ax.set_title("Farneback Flow (Angular Magnitude, rad/s)")
farneback_ax.set_xlabel("Pixel X Coordinate")
farneback_ax.set_ylabel("Pixel Y Coordinate")
farneback_ax.invert_yaxis()

diff_ax.set_title("Difference in Angular Magnitude (rad/s)")
diff_ax.set_xlabel("Pixel X Coordinate")
diff_ax.set_ylabel("Pixel Y Coordinate")
diff_ax.invert_yaxis()



def calculate_optical_flow():
    # Capture two consecutive images
    prev_image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    prev_img = np.frombuffer(prev_image.image_data_uint8, dtype=np.uint8).reshape(prev_image.height, prev_image.width, 3)
    time.sleep(1 / fps)  # Wait for the next frame
    curr_image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    curr_img = np.frombuffer(curr_image.image_data_uint8, dtype=np.uint8).reshape(curr_image.height, curr_image.width, 3)

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Calculate Farneback optical flow
    farneback_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                  pyr_scale=0.5, levels=3, winsize=9,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Retrieve depth map
    depth_image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)])[0]
    depth_map = np.array(depth_image.image_data_float, dtype=np.float32).reshape(depth_image.height, depth_image.width)

    # Get real kinematic data
    kinematics = client.simGetGroundTruthKinematics()
    T = np.array([
                    -kinematics.linear_velocity.y_val,
                    -kinematics.linear_velocity.z_val,
                    kinematics.linear_velocity.x_val
                  
                ])
    omega = np.array([-kinematics.angular_velocity.y_val,
                      -kinematics.angular_velocity.z_val,
                      kinematics.angular_velocity.x_val])
    

    # Get image dimensions and pixel coordinates
    height, width = prev_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords - width // 2
    y_coords = y_coords - height // 2

 

    # Extract translational and rotational velocities
    T_x, T_y, T_z = T
    Omega_x, Omega_y, Omega_z = omega

    # # Compute real-world displacements
    # u = (-focal_length_px / depth_map) * (T_x + Omega_y * depth_map - Omega_z * y_coords)
    # v = (-focal_length_px / depth_map) * (T_y + Omega_z * x_coords - Omega_x * depth_map)


    ''' test Matt's geometric OF'''
    X_dot, Y_dot, Z_dot = T#sideways_velocity, vertical_velocity, normal_velocity
    p, q, r = omega
    frame_time = 0.153   # Frame duration (10 FPS == 0.1 frame_time)
  
    # Combine velocities and rotations into a single state vector
    state_vector = np.array([X_dot, Y_dot, Z_dot, p, q, r])

    Fx = Fy = focal_length_px
    Z = depth_map

    # Build the matrix
    Z_height, Z_width = Z.shape
    #print(f"Z_height:{Z_height};Z_width:{Z_width}")
    OF_list = []
    Qu_list = []
    Qv_list = []
    geometric_flow = np.zeros((Z_height,Z_width,2))
    # Define the center of the image
    u_center = Z_width // 2
    v_center = Z_height // 2
    for v in range(Z_height):
        for u in range(Z_width):
              # Shift u and v so that the origin is at the center
            u_shifted = -(u - u_center) #?
            v_shifted = v_center - v 
            # print(f"Orignial pixel position x: {u}; y: {v}; After transform x: {u_shifted}; y:{v_shifted}") 
            z = Z[v, u]
            a_row1 = [Fx / z, 0, -u_shifted / z, -u_shifted * v_shifted / Fx, -Fx - (u_shifted**2) / Fx, -v_shifted]  
            a_row2 = [0, Fy / z, -v_shifted / z, -Fy - (v_shifted**2) / Fy, -u_shifted * v_shifted / Fy, u_shifted]
            # a_row1 = [Fx / z, 0, -u / z, -u * v / Fx, -Fx - (u**2) / Fx, -v]
            # a_row2 = [0, Fy / z, -v / z, -Fy - (v**2) / Fy, -u * v / Fy, u]
            Qu = np.matmul(a_row1,state_vector.T)
            Qv = np.matmul(a_row2,state_vector.T)
            # print(f"Qu: {Qu},Qv:{Qv}")
            geometric_flow[v, u, 0] = Qu #/ (Fx * 0.88) 
            geometric_flow[v, u, 1] = Qv #/ (Fy * 0.88)
            Qu_list.append(Qu)
            Qv_list.append(Qv)
            OF_list.append([Qu,Qv])  
           
  
    '''end test'''
    

    # Convert to angular velocities
    geom_angular_x = geometric_flow[..., 0] / focal_length_px
    geom_angular_y = geometric_flow[..., 1] / focal_length_px

    # geom_angular_x = u / focal_length_px
    # geom_angular_y = v / focal_length_px



    # Convert Farneback flow to angular velocity (rad/s)
    farneback_angular_x = farneback_flow[..., 0] / focal_length_px * fps
    farneback_angular_y = farneback_flow[..., 1] / focal_length_px * fps

    return geom_angular_x, geom_angular_y, farneback_angular_x, farneback_angular_y

def data_logging(geom_angular_x, geom_angular_y, farneback_angular_x, farneback_angular_y):


     # Calculate differences for comparison
    diff_x = np.abs(geom_angular_x - farneback_angular_x)
    diff_y = np.abs(geom_angular_y - farneback_angular_y)


    # Calculate angular velocity magnitudes
    geom_magnitude = np.sqrt(geom_angular_x**2 + geom_angular_y**2)  # Replace with updated geom_angular_x, geom_angular_y
    farneback_magnitude = np.sqrt(farneback_angular_x**2 + farneback_angular_y**2)  # Replace with updated farneback_angular_x, farneback_angular_y
    diff_magnitude = np.abs(geom_magnitude - farneback_magnitude)

    # Update the data dynamically
    geom_plot.set_data(geom_magnitude)
    farneback_plot.set_data(farneback_magnitude)
    diff_plot.set_data(diff_magnitude)

    # Update colorbar limits dynamically
    geom_plot.set_clim(vmin=np.min(geom_magnitude), vmax=np.max(geom_magnitude))
    farneback_plot.set_clim(vmin=np.min(farneback_magnitude), vmax=np.max(farneback_magnitude))
    diff_plot.set_clim(vmin=np.min(diff_magnitude), vmax=np.max(diff_magnitude))

    # Pause briefly to display the updated frame
    plt.pause(0.01)
  

start_time = time.time()
while time.time() - start_time < duration:
    # Command movement for a short interval
    start_time_loop = time.time()
    client.moveByVelocityAsync(velocity, 0, 0, 1)
    time.sleep(interval)


    # Calculate flows
    geom_angular_x, geom_angular_y, farneback_angular_x, farneback_angular_y = calculate_optical_flow()

    data_logging(geom_angular_x, geom_angular_y, farneback_angular_x, farneback_angular_y)
    end_time_loop = time.time()

    # Calculate frame duration (in seconds)
    frame_duration = end_time_loop - start_time_loop
    print(f"Frame duration: {frame_duration} seconds")

   


client.hoverAsync().join()
client.armDisarm(False)
client.enableApiControl(False)