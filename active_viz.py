import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches

# Parameters
num_particles = 100
num_steps = 100
step_size = 0.1
circle_radius = 0.4

# Generate initial positions
np.random.seed(0)
positions = np.random.randn(num_particles, 2)

# Generate random velocities
velocities = np.random.randn(num_particles, 2)

# Create figure and axis
fig = plt.figure()
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1, aspect=1)

# Set axis limits
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Create scatter plot for particles
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=300, facecolor='gray', edgecolor='black')

# Create arrow patches for particle directions
arrows = []
for _ in range(num_particles):
    arrow = patches.FancyArrow(0, 0, 0, 0, width=0.1, color='black', head_length=1, head_width=1)
    ax.add_patch(arrow)
    arrows.append(arrow)

# Update function for animation
def update(frame):
    # Move particles
    positions[:, 0] += velocities[:, 0] * step_size
    positions[:, 1] += velocities[:, 1] * step_size

    # Reflect particles at the boundaries
    for i in range(num_particles):
        if abs(positions[i, 0]) > 10:
            velocities[i, 0] *= -1
        if abs(positions[i, 1]) > 10:
            velocities[i, 1] *= -1

    # Update scatter plot data
    scatter.set_offsets(positions)

    # Update arrow positions and directions
    for i in range(num_particles):
        x, y = positions[i]
        u, v = velocities[i]
        angle = np.arctan2(v, u)

        # Remove old arrow
        arrow = arrows[i]
        arrow.remove()
        
        speed = np.sqrt(u**2 + v**2)
        # Calculate arrow length based on velocity magnitude
        length = np.sqrt(u**2 + v**2)

        # Create new arrow
        arrow = patches.FancyArrow(x-u/speed/5, y-v/speed/5, u/length/speed/300,
                                   v/length/speed/300, width=0.1, color='lightsalmon',
                                   head_length=0.4,head_width=0.2)
        ax.add_patch(arrow)
        arrows[i] = arrow

    # Return the updated scatter plot and arrows
    return scatter, *arrows

# Animate the figure
animation = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)
animation.save('asd.mp4')


# Show the plot 
plt.show()