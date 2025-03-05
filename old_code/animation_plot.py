import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv("average_optical_flow.csv", header=None)

# Select every alternate row for fb and geo
alternative_rows_fb = data.iloc[::2, 10:310]
alternative_rows_geo = data.iloc[1::2, 10:310]

# Set up the x-axis as the column indices
x_axis = np.arange(alternative_rows_fb.shape[1])

# Prepare the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# First plot: FB Data and Geo Data
line_fb, = ax1.plot([], [], lw=2, label="FB Data", color="blue")
line_geo, = ax1.plot([], [], lw=2, label="Geo Data", color="orange")
ax1.set_xlim(0, len(x_axis) - 1)
# ax1.set_ylim(
#     min(alternative_rows_fb.values.min(), alternative_rows_geo.values.min()),
#     max(alternative_rows_fb.values.max(), alternative_rows_geo.values.max()),
# )
ax1.set_ylim(
    0,
    0.5
)
ax1.set_title("FB Data and Geo Data")
ax1.set_xlabel("Column Index")
ax1.set_ylabel("Value")
ax1.legend()

# Second plot: Difference (FB - Geo)
line_diff, = ax2.plot([], [], lw=2, label="Difference (FB - Geo)", color="green")
ax2.set_xlim(0, len(x_axis) - 1)
ax2.set_ylim(
    (alternative_rows_fb.values - alternative_rows_geo.values).min(),
    (alternative_rows_fb.values - alternative_rows_geo.values).max(),
)
ax2.set_title("Difference (FB - Geo)")
ax2.set_xlabel("Column Index")
ax2.set_ylabel("Difference")
ax2.legend()

# Add text for frame number in both plots
frame_text_ax1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, color='red')
frame_text_ax2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12, color='red')

# Initialization function
def init():
    line_fb.set_data([], [])
    line_geo.set_data([], [])
    line_diff.set_data([], [])
    frame_text_ax1.set_text('')
    frame_text_ax2.set_text('')
    return line_fb, line_geo, line_diff, frame_text_ax1, frame_text_ax2

# Animation function
def animate(i):
    y_fb = alternative_rows_fb.iloc[i].values
    y_geo = alternative_rows_geo.iloc[i].values
    y_diff = y_fb - y_geo
    line_fb.set_data(x_axis, y_fb)
    line_geo.set_data(x_axis, y_geo)
    line_diff.set_data(x_axis, y_diff)
    frame_text_ax1.set_text(f'Frame: {i}')
    frame_text_ax2.set_text(f'Frame: {i}')
    return line_fb, line_geo, line_diff, frame_text_ax1, frame_text_ax2

# Create the animation
frames = min(len(alternative_rows_fb), len(alternative_rows_geo))  # Match frame count
anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True, repeat=False)

# Show the animation
plt.tight_layout()

fb_data = data.iloc[100::2, 10:310]
geo_data = data.iloc[101::2, 10:310]

percentage_error = np.abs((fb_data.values - geo_data.values) / geo_data.values) * 100

# For each frame (row), find the maximum percentage error
max_percentage_error_per_frame = np.max(percentage_error, axis=1)

# The x-axis represents the frame number (0, 1, 2, ...)
frames = np.arange(fb_data.shape[0])

# Create a static plot of the maximum percentage error per frame
plt.figure(figsize=(10, 6))
plt.plot(frames, max_percentage_error_per_frame, marker='o', linestyle='-', color='red')
plt.xlabel("Frame Number")
plt.ylabel("Max Percentage Error (%)")
plt.title("Maximum Percentage Error Between FB and Geo Data Over Frames")
plt.grid(True)
plt.tight_layout()
plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import pandas as pd
# import numpy as np

# # Load your dataset
# data = pd.read_csv("average_optical_flow.csv", header=None)

# # Select every alternate row for fb and geo
# alternative_rows_fb = data.iloc[::2, :]
# alternative_rows_geo = data.iloc[1::2, :]

# # Set up the x-axis as the column indices
# x_axis = np.arange(alternative_rows_fb.shape[1])

# # Prepare the figure and axis
# fig, ax = plt.subplots()
# line_fb, = ax.plot([], [], lw=2, label="FB Data", color="blue")
# line_geo, = ax.plot([], [], lw=2, label="Geo Data", color="orange")
# ax.set_xlim(0, len(x_axis) - 1)
# ax.set_ylim(
#     min(alternative_rows_fb.values.min(), alternative_rows_geo.values.min()),
#     max(alternative_rows_fb.values.max(), alternative_rows_geo.values.max()),
# )
# ax.set_title("Line Draft Animation")
# ax.set_xlabel("Column Index")
# ax.set_ylabel("Value")
# ax.legend()

# # Initialization function
# def init():
#     line_fb.set_data([], [])
#     line_geo.set_data([], [])
#     return line_fb, line_geo

# # Animation function
# def animate(i):
#     y_fb = alternative_rows_fb.iloc[i].values
#     y_geo = alternative_rows_geo.iloc[i].values
#     line_fb.set_data(x_axis, y_fb)
#     line_geo.set_data(x_axis, y_geo)
#     return line_fb, line_geo



# # Create the animation
# frames = min(len(alternative_rows_fb), len(alternative_rows_geo))  # Match frame count
# anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True,repeat=False)

# # Show the animation
# plt.show()
