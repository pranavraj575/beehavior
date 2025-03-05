import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

# Load datasets
gmflow_data = pd.read_csv("efRAFT_avg_u.csv", header=None)

# Convert GMFlow values to absolute values
gmflow_data = np.abs(gmflow_data)


# Resize GMFlow data to match the same width (columns 10:310)
alternative_rows_gmflow = gmflow_data.iloc[:, 10:310]  # Select columns 10:310

# Check average values of the datasets
mean_gmflow = np.mean(alternative_rows_gmflow.values)


print(f"Mean GMFlow (before conversion): {mean_gmflow:.6f}")

# Assume GMFlow is in pixels/frame if its values are significantly larger

# Define known parameters for conversion
FOV_radians = np.deg2rad(60)  # Example: Convert 90° FOV to radians
image_width = 320  # Adjust based on actual width of images
frame_interval = 0.0184 # Assuming 30 FPS

# Convert GMFlow from pixels/frame to radians/second
alternative_rows_gmflow = (alternative_rows_gmflow / image_width) * (FOV_radians / frame_interval)

mean_gmflow = np.mean(alternative_rows_gmflow.values)  # Recalculate after conversion
print(f"Mean GMFlow (after conversion to rad/s): {mean_gmflow:.6f}")

# Set up the x-axis as the column indices
x_axis = np.arange(alternative_rows_gmflow.shape[1])  # Should be range(300) since 10:310 gives 300 values

# Prepare the figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Initialize lines for animation
line_gmflow, = ax1.plot([], [], lw=2, label="efRAFT Data (Converted)", color="red")

# Set axis limits
ax1.set_xlim(0, len(x_axis) - 1)
ax1.set_ylim(
    0,
   alternative_rows_gmflow.values.max()*1.1
)
ax1.set_title("efRAFT Optical Flow Comparison")
ax1.set_xlabel("Column Index")
ax1.set_ylabel("Value (rad/s)")
ax1.legend()

# Add text for frame number
frame_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, color='red')

# Initialization function
def init():
    line_gmflow.set_data([], [])
    frame_text.set_text('')
    return line_gmflow, frame_text

# Animation function
def animate(i):
    y_gmflow = alternative_rows_gmflow.iloc[i].values  # Resized GMFlow absolute values

    line_gmflow.set_data(x_axis, y_gmflow)

    frame_text.set_text(f'Frame: {i}')
    return line_gmflow, frame_text

# Create the animation
frames = len(alternative_rows_gmflow)  # Match frame count
anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True, repeat=False)

# Show the animation
plt.tight_layout()
plt.show()
