#importing libraries

import numpy as np
import matplotlib.pyplot as plt

# 1) declaring variables:

N_particles = 150  #no. of particles
G=1 # gravitational constant
dt=0.01 # time step
num_steps=100 # number of time steps

#high density initial conditions
np.random.seed(101)
positions = np.random.normal(0, 2, (N_particles, 3)) #shape: (N_particles 3 for x, y, z)
velocities = np.random.normal(0, 0.5, (N_particles, 3)) #random velocities for (N_particles 3 for vx, vy, vz)

#as initial condition of 0 would collapse
softening=0.1

# 2) gravitational force as vectors:

def compute_gravitational_accelerations(pos):

    """
    Computes acceleration for all particles using vectorized numpy operations.
    """

    #to get 3D matrices
    dx = pos[:, 0:1].T - pos[:, 0:1]

    dy = pos[:, 1:2].T - pos[:, 1:2]

    dz = pos[:, 2:3].T - pos[:, 2:3]

    # matrix of distances sqaured
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)

    ax = G * (dx * inv_r3) @ np.ones(N_particles)
    ay = G * (dy * inv_r3) @ np.ones(N_particles)
    az = G * (dz * inv_r3) @ np.ones(N_particles)

    return np.vstack((ax, ay, az)).T

#3) updating positions and velocities using leapfrog integration:

history = [] 

#kick drift kick
for step in range(num_steps):
    acc=compute_gravitational_accelerations(positions)
    velocities = velocities + acc * dt
    positions = positions + velocities * dt

#a snapshot every 20 steps

    if step % 20 == 0 or step == num_steps - 1:
        history.append(positions.copy())


#plotting
stages = [0, 1, 2, 4]  # stages to visualize
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), subplot_kw={'projection': '3d'})
titles= ['Initial Fluctuations', 'Early Growth', 'Void Formation', 'Cosmic Web']

for idx, stage_num in enumerate(stages):

    ax = axes[idx]

    pos_snapshot = history[stage_num]

    
    # Draw dark matter particles

    ax.scatter(pos_snapshot[:, 0], pos_snapshot[:, 1], pos_snapshot[:, 2],
               s=4, color='indigo', alpha=0.6)

    
    ax.set_title(titles[idx], fontsize=11)

    ax.set_xlim(-10, 10)

    ax.set_ylim(-10, 10)

    ax.set_zlim(-10, 10)

    ax.axis('off')  # Hide axes box for clean cosmic look



plt.suptitle("Evolution of Dark Matter Large-Scale Structure (N-Body Simulation)", fontsize=14)

plt.tight_layout()

plt.show()