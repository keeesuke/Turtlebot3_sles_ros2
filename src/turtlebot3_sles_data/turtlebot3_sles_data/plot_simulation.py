import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data
plot_file = os.path.expanduser("~/plot_data_lowlvlctrl.npz")
failure_file = os.path.expanduser('~/mpc_failure_data.npz')
data = np.load(plot_file, allow_pickle=True)
data_failure = np.load(failure_file)
plot_data = data["data"]

# Extract data
Xopt_0 = np.array([entry["Xopt_0"] for entry in plot_data])
x0 = np.array([entry["x0"] for entry in plot_data])
errors = np.array([entry["errors"] for entry in plot_data])
v_cmd = np.array([entry["v_cmd"] for entry in plot_data])
omega_cmd = np.array([entry["omega_cmd"] for entry in plot_data])

# Time steps
time_steps = np.arange(len(plot_data))

grid_size = 0.02
n = 4/0.02 #occ dimension
origin = (-2,-2)
xi = (np.arange(n)+0.5)*grid_size + origin[0]
yi = (np.arange(n)+0.5)*grid_size + origin[1]
Xg, Yg = np.meshgrid(xi, yi)

start  = x0[0,:]        # shape (5,) 
target = data_failure['goal']      # shape (5,)
print('start is ', start)
print('target is ', target)

raw_occ = data_failure['occ']      # integer occupancy grid
occ_bool = raw_occ >= 50 # 0 (white) for free, 1 (black) for occupied
robot_size = 0.25
traj = x0
traj_ref = Xopt_0

# 1) Plot the trajectory over occ_bool
plt.figure(figsize=(6,6))
plt.pcolormesh(
    Xg, Yg, occ_bool,
    shading='auto',
    cmap='gray',
    alpha=0.5
)
plt.plot(traj_ref[:,0], traj_ref[:,1], 'r:', lw=1.5, label='Reference Trajectory')
plt.plot(traj[:,0], traj[:,1], 'b--', lw=2, label='MPC Trajectory')
plt.plot(start[0], start[1], 'go')
# plt.plot(target[0], target[1], 'rx')

# Plot the target position as a cross
plt.plot(target[0], target[1], 'rx', markersize=10, label='Target Position')
plt.plot(x0[0,0], x0[0,1], 'go', markersize=10, label='Initial Position')

# Plot each trajectory state as a circle
for state in traj:
    x, y = state[0], state[1]
    circle = plt.Circle((x, y), radius=robot_size / 2, color='cyan', alpha=0.3)
    plt.gca().add_patch(circle)  # Add the circle to the current axes
# Final state
x, y = traj[-1, 0], traj[-1, 1]
circle = plt.Circle((x, y), radius=robot_size / 2, color='b', alpha=0.8)
plt.gca().add_patch(circle)

xmin, ymin = origin
xmax = xmin + occ_bool.shape[1] * grid_size
ymax = ymin + occ_bool.shape[0] * grid_size
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.axis('equal'); plt.grid(True)
plt.title("MPC w/ SDF Collision Avoidance")
plt.legend(loc='lower right')

# Show all plots
plt.show()