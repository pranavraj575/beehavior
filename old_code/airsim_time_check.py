import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

# Get initial timestamp
state = client.getMultirotorState()
initial_time = state.timestamp

print(f"Initial simulation time: {initial_time}")

# Pause the simulation
client.simPause(True)
time.sleep(2)  # Wait 2 seconds in real-world time

# Get timestamp while paused
state = client.getMultirotorState()
paused_time = state.timestamp
print(f"Simulation time while paused: {paused_time}")

# Resume and check time again
client.simPause(False)
time.sleep(1)

state = client.getMultirotorState()
resumed_time = state.timestamp
print(f"Simulation time after resuming: {resumed_time}")
